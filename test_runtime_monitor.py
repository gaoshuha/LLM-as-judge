"""Local, zero-cost tests for streaming progress and hard request cancellation."""

from __future__ import annotations

import json
import io
import multiprocessing as mp
import threading
import time
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import position_bias_experiment as experiment


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # noqa: ANN001
        pass


class StreamingHandler(BaseHTTPRequestHandler):
    initial_delay = 0.05

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        time.sleep(self.initial_delay)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for text in ('{"winner":', '"A","reason":"ok"}'):
            event = {"choices": [{"delta": {"content": text}}]}
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args) -> None:  # noqa: ANN002
        pass


class SlowHandler(StreamingHandler):
    initial_delay = 5.0


def serve(handler: type[BaseHTTPRequestHandler]) -> QuietServer:
    server = QuietServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    experiment.JUDGE_API_KEY = "local-test-key"
    experiment.MAX_RETRIES = 1
    experiment.STATUS_REFRESH_SECONDS = 0.05

    normal = serve(StreamingHandler)
    experiment.JUDGE_BASE_URL = f"http://127.0.0.1:{normal.server_port}/v1"
    normal_console = io.StringIO()
    with redirect_stdout(normal_console):
        raw = experiment.api_chat("test prompt", "90", "baseline", "forward", 5, 0)
    assert json.loads(raw)["winner"] == "A"
    assert normal_console.getvalue().count("API调用 |") == 1
    assert "API状态 |" not in normal_console.getvalue()
    normal.shutdown()
    normal.server_close()

    slow = serve(SlowHandler)
    experiment.JUDGE_BASE_URL = f"http://127.0.0.1:{slow.server_port}/v1"
    started = time.monotonic()
    slow_console = io.StringIO()
    with redirect_stdout(slow_console):
        try:
            experiment.api_chat("test prompt", "90", "baseline", "forward", 0.35, 0)
            raise AssertionError("Expected automatic skip")
        except experiment.SkipQuestion:
            pass
    assert slow_console.getvalue().count("API调用 |") == 1
    elapsed = time.monotonic() - started
    slow.shutdown()
    slow.server_close()
    assert elapsed < 2.5, f"Cancellation was not immediate: {elapsed:.2f}s"
    assert not mp.active_children(), "An API child process was left behind"
    timeout_server = serve(SlowHandler)
    experiment.JUDGE_BASE_URL = f"http://127.0.0.1:{timeout_server.server_port}/v1"
    timeout_console = io.StringIO()
    with redirect_stdout(timeout_console):
        try:
            experiment.api_chat("test prompt", "90", "baseline", "forward", 5, 0.25)
            raise AssertionError("Expected first-output timeout")
        except TimeoutError:
            pass
    timeout_server.shutdown()
    timeout_server.server_close()
    assert timeout_console.getvalue().count("API调用 |") == 1
    assert "[TIMEOUT]" in timeout_console.getvalue()
    assert not mp.active_children(), "First-output timeout left an API child process"
    print("STREAM_OK; HARD_CANCEL_OK; FIRST_OUTPUT_TIMEOUT_OK; NO_ORPHAN_PROCESS; STATIC_ONE_LINE_OK")


if __name__ == "__main__":
    mp.freeze_support()
    main()
