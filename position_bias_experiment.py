#!/usr/bin/env python3
"""Reproducible LLM-as-a-Judge position-bias experiment for MT-Bench.

The candidate trajectories are fixed in answers.jsonl. Ground-truth labels are
loaded only after all judge calls finish and are never included in prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


# =========================================================
# Step 0: 在这里输入 LLM Judge 的 API Key
# JUDGE_API_KEY = "PASTE_YOUR_API_KEY_HERE"
# JUDGE_MODEL_NAME = "deepseek-v4-flash"
# JUDGE_BASE_URL = "https://api.deepseek.com/v1"

JUDGE_API_KEY = "PASTE_YOUR_API_KEY_HERE"
JUDGE_MODEL_NAME = "google/gemma-4-31b-it"
JUDGE_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


# Reproducibility and reliability settings
TEMPERATURE = 0
SEED = 0
MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 120
RETRY_BASE_SECONDS = 2
STATUS_REFRESH_SECONDS = 1.0
STALL_WARNING_SECONDS = 30.0
DEFAULT_MAX_CALL_SECONDS = 300.0
DEFAULT_FIRST_OUTPUT_TIMEOUT_SECONDS = 30.0
DEFAULT_API_TEST_TIMEOUT_SECONDS = 15.0


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
QUESTIONS_PATH = ROOT / "question.jsonl"
ANSWERS_PATH = ROOT / "answers_v1.jsonl"
GROUND_TRUTH_PATH = ROOT / "ground_truth.json"
PROMPTS_PATH = ROOT / "judge_prompts.jsonl"
OUTPUTS_PATH = ROOT / "judge_outputs.jsonl"
MAPPED_PATH = ROOT / "mapped_results.csv"
METRICS_PATH = ROOT / "metrics_summary.csv"
REPORT_PATH = ROOT / "report.md"
FAILURES_PATH = ROOT / "parse_failures.jsonl"

SYSTEM_PROMPT = """You are an impartial evaluator. Compare two anonymous responses to the same user conversation. Judge correctness, relevance, instruction-following, clarity, and completeness. Do not infer authorship or quality from response position, length, style, or the labels A and B. Treat the two positions symmetrically. A tie is allowed only when their overall quality is genuinely indistinguishable. Return valid JSON only."""


def configure_answer_version(version: str, run_name: str | None = None) -> None:
    """Select an answer set and keep checkpoints/results isolated by version.

    V3 = verbosity-bias set: candidate_2 is the padded (>=2x length) rewrite
    of the V2 weak answer, built once by build_answers_v3.py.
    V4 = surface-persuasion set: candidate_2 is the unchanged V2 weak answer
    wrapped in fixed authority, consensus, style, and compassion cues.
    """
    global ANSWERS_PATH, PROMPTS_PATH, OUTPUTS_PATH, MAPPED_PATH
    global METRICS_PATH, REPORT_PATH, FAILURES_PATH
    ANSWERS_PATH = ROOT / f"answers_v{version}.jsonl"
    if run_name is None:
        # Keep the original single-version paths compatible with old commands
        # and existing checkpoints.
        suffix = "" if version == "1" else f"_v{version}"
        output_dir = ROOT
        PROMPTS_PATH = output_dir / f"judge_prompts{suffix}.jsonl"
        OUTPUTS_PATH = output_dir / f"judge_outputs{suffix}.jsonl"
        MAPPED_PATH = output_dir / f"mapped_results{suffix}.csv"
        METRICS_PATH = output_dir / f"metrics_summary{suffix}.csv"
        REPORT_PATH = output_dir / f"report{suffix}.md"
        FAILURES_PATH = output_dir / f"parse_failures{suffix}.jsonl"
        return

    if not re.fullmatch(r"V[1-4]R[1-9][0-9]*", run_name):
        raise ValueError(f"Invalid experiment run name: {run_name!r}")
    output_dir = RESULTS_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    PROMPTS_PATH = output_dir / "judge_prompts.jsonl"
    OUTPUTS_PATH = output_dir / "judge_outputs.jsonl"
    MAPPED_PATH = output_dir / "mapped_results.csv"
    METRICS_PATH = output_dir / "metrics_summary.csv"
    REPORT_PATH = output_dir / "report.md"
    FAILURES_PATH = output_dir / "parse_failures.jsonl"


def parse_answer_versions(value: str, parser: argparse.ArgumentParser) -> list[str]:
    """Parse a comma/space separated version list while preserving its order."""
    parts = [part.upper().removeprefix("V") for part in re.split(r"[,，\s]+", value.strip()) if part]
    if not parts or any(part not in {"1", "2", "3", "4"} for part in parts):
        parser.error("--answer-versions 必须是 1、2、3、4 的组合，例如 2,3,4")
    return list(dict.fromkeys(parts))


def choose_experiment_plan(args: argparse.Namespace,
                           parser: argparse.ArgumentParser) -> tuple[list[str], int, bool]:
    """Return selected versions, repeat count, and whether named run folders are used."""
    if args.answer_version and args.answer_versions:
        parser.error("--answer-version 和 --answer-versions 不能同时使用")
    if args.rounds is not None and args.rounds < 1:
        parser.error("--rounds 必须大于或等于 1")

    if args.answer_versions:
        return parse_answer_versions(args.answer_versions, parser), args.rounds or 1, True
    if args.answer_version:
        # An explicitly supplied --rounds opts into the new VxRy folder layout;
        # the old command without --rounds retains its original paths.
        return [args.answer_version], args.rounds or 1, args.rounds is not None
    if not sys.stdin.isatty():
        parser.error("非交互环境必须指定 --answer-version 或 --answer-versions")

    print("\n请选择要运行的候选回答版本。")
    print("可输入一个或多个版本，例如 2,3,4：")
    while True:
        try:
            raw_versions = input("版本：").strip()
        except EOFError:
            parser.error("无法读取交互输入，请使用 --answer-versions 指定版本")
        parts = [
            part.upper().removeprefix("V")
            for part in re.split(r"[,，\s]+", raw_versions)
            if part
        ]
        if parts and all(part in {"1", "2", "3", "4"} for part in parts):
            versions = list(dict.fromkeys(parts))
            break
        print("输入无效，请输入 1、2、3、4 的组合，例如 2,3,4。")
    while True:
        try:
            raw_rounds = input("每个版本进行几轮实验：").strip()
        except EOFError:
            parser.error("无法读取交互输入，请使用 --rounds 指定轮数")
        if raw_rounds.isdigit() and int(raw_rounds) >= 1:
            return versions, int(raw_rounds), True
        print("输入无效，请输入大于或等于 1 的整数。")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def validate_inputs(questions: list[dict[str, Any]], answers: list[dict[str, Any]],
                    truth: list[dict[str, Any]]) -> None:
    if len(questions) != 80:
        raise ValueError(f"Expected 80 questions, found {len(questions)}")
    q_by_id = {str(x["question_id"]): x for x in questions}
    a_by_id = {str(x["question_id"]): x for x in answers}
    t_by_id = {str(x["question_id"]): x for x in truth}
    if set(q_by_id) != set(a_by_id) or set(q_by_id) != set(t_by_id):
        raise ValueError("question_id sets differ among input files")
    for qid, q in q_by_id.items():
        turns = q.get("turns")
        a = a_by_id[qid]
        if not isinstance(turns, list) or len(turns) != 2:
            raise ValueError(f"Question {qid} must contain exactly two turns")
        for candidate in ("candidate_1", "candidate_2"):
            if not isinstance(a.get(candidate), list) or len(a[candidate]) != 2:
                raise ValueError(f"{qid}.{candidate} must contain two responses")
        t = t_by_id[qid]
        if {t.get("strong_candidate"), t.get("weak_candidate")} != {"candidate_1", "candidate_2"}:
            raise ValueError(f"Invalid ground truth for {qid}")


def render_conversation(question_turns: list[str], a_turns: list[str], b_turns: list[str]) -> str:
    parts: list[str] = []
    for idx, (question, a, b) in enumerate(zip(question_turns, a_turns, b_turns), 1):
        parts.extend([
            f"User turn {idx}:\n{question}",
            f"Response A, turn {idx}:\n{a}",
            f"Response B, turn {idx}:\n{b}",
        ])
    return "\n\n".join(parts)


def build_prompt(question: dict[str, Any], answer: dict[str, Any], order: str,
                 method: str) -> str:
    if order == "forward":
        a_turns, b_turns = answer["candidate_1"], answer["candidate_2"]
    else:
        a_turns, b_turns = answer["candidate_2"], answer["candidate_1"]
    conversation = render_conversation(question["turns"], a_turns, b_turns)
    if method == "baseline":
        instruction = """Evaluate both response trajectories holistically. Return exactly one JSON object in this schema:
{"winner":"A|B|tie","reason":"brief justification"}
The winner value must be A, B, or tie."""
    else:
        instruction = """Reason before deciding: compare the two response trajectories, identify material advantages and limitations of each, and only then decide. Return exactly one JSON object in this schema:
{"reason":"comparison including advantages and limitations of both responses","final_winner":"A|B|tie"}
The final_winner value must be A, B, or tie."""
    return f"{instruction}\n\n{conversation}"


class SkipQuestion(Exception):
    """Raised when the user or watchdog requests skipping the current question."""


def _estimate_input_chars(prompt: str) -> int:
    return len(SYSTEM_PROMPT) + len(prompt)


def _chat_completions_url(base_url: str) -> str:
    """Accept either an API root (/v1) or a full chat-completions URL."""
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def test_api_connection(timeout_seconds: float = DEFAULT_API_TEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Make one minimal request and verify the configured Judge API is usable."""
    if not JUDGE_API_KEY or JUDGE_API_KEY == "PASTE_YOUR_JUDGE_API_KEY_HERE":
        raise RuntimeError("JUDGE_API_KEY is not configured")

    payload = {
        "model": JUDGE_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": 'Reply exactly with {"ok":true}.'},
        ],
        "temperature": TEMPERATURE,
        "seed": SEED,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    request = urllib.request.Request(
        _chat_completions_url(JUDGE_BASE_URL),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {JUDGE_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}{suffix}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"connection or response error: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("API responded, but did not return valid JSON message content") from exc
    if result.get("ok") is not True:
        raise RuntimeError(f"API responded, but the test content was unexpected: {content[:200]}")
    return {
        "model": data.get("model", JUDGE_MODEL_NAME),
        "elapsed_seconds": time.monotonic() - started,
    }


def _api_chat_process(prompt: str, api_config: dict[str, Any], result_queue: Any,
                      progress_queue: Any) -> None:
    """Isolated API process: it can be terminated without leaving an orphan request."""
    if api_config["api_key"] == "PASTE_YOUR_JUDGE_API_KEY_HERE":
        result_queue.put(("error",
                          "Set JUDGE_API_KEY at the top of this file, or run with --mock."))
        return
    url = _chat_completions_url(api_config["base_url"])
    payload = {
        "model": api_config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": api_config["temperature"],
        "seed": api_config["seed"],
        "response_format": {"type": "json_object"},
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_config['api_key']}",
                 "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(api_config["max_retries"]):
        progress_queue.put(("update", {
            "attempt": attempt + 1, "phase": "connecting",
            "attempt_started_at": time.monotonic(), "first_output_at": None,
            "last_output_at": None, "output_chars": 0, "retry_error": "",
        }))
        try:
            with urllib.request.urlopen(
                    request, timeout=api_config["request_timeout_seconds"]) as response:
                progress_queue.put(("update", {"phase": "waiting_first_output"}))
                chunks: list[str] = []
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line == "data: [DONE]":
                        break
                    encoded = line[5:].strip() if line.startswith("data:") else line
                    try:
                        event = json.loads(encoded)
                    except json.JSONDecodeError:
                        continue
                    # Compatibility fallback for a provider that ignores stream=True.
                    message = event.get("choices", [{}])[0].get("message", {})
                    if message.get("content") is not None:
                        content = str(message["content"])
                        chunks.append(content)
                        progress_queue.put(("output", len(content), time.monotonic()))
                        continue
                    delta = event.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    if content:
                        chunks.append(str(content))
                    # Hidden reasoning is counted for liveness/speed but not added to JSON output.
                    progress_queue.put(("output", len(str(content) + str(reasoning)),
                                        time.monotonic()))
                result = "".join(chunks)
                if not result:
                    raise ValueError("Streaming response contained no final content")
            progress_queue.put(("update", {"phase": "done"}))
            result_queue.put(("result", result))
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError,
                json.JSONDecodeError, OSError, ValueError) as exc:
            last_error = exc
            progress_queue.put(("update", {"phase": "retry_wait",
                                            "retry_error": repr(exc)}))
            if attempt + 1 < api_config["max_retries"]:
                time.sleep(api_config["retry_base_seconds"] * (2 ** attempt))
    result_queue.put(("error",
                      f"Judge API failed after {api_config['max_retries']} attempts: {last_error}"))


def _skip_key_pressed() -> bool:
    """Non-blocking keyboard check. Interactive Windows terminals support S to skip."""
    if not sys.stdin.isatty():
        return False
    if os.name == "nt":
        try:
            import msvcrt
            while msvcrt.kbhit():
                if msvcrt.getwch().lower() == "s":
                    return True
        except (ImportError, OSError):
            return False
    return False


def _progress_line(state: dict[str, Any], now: float, previous_output_chars: int,
                   previous_time: float) -> tuple[str, int]:
    snapshot = dict(state)
    elapsed = now - snapshot["started_at"]
    first = snapshot["first_output_at"]
    last = snapshot["last_output_at"]
    current_chars_per_second = max(0, snapshot["output_chars"] - previous_output_chars) / max(
        now - previous_time, 1e-9)
    input_speed = 0.0 if first is None else snapshot["input_chars"] / max(
        first - snapshot["attempt_started_at"], 1e-9)
    if snapshot["phase"] == "retry_wait":
        status = "重试等待"
    elif first is None:
        attempt_elapsed = now - snapshot["attempt_started_at"]
        status = "疑似停滞" if attempt_elapsed >= STALL_WARNING_SECONDS else "等待首输出"
    elif last is not None and now - last >= STALL_WARNING_SECONDS:
        status = "疑似停滞"
    else:
        status = "正在输出"
    line = (
        f"API状态 | 题目 {snapshot['question_id']} | {snapshot['method']}/{snapshot['order']} "
        f"| {status} | {elapsed:6.1f}s | 第 {snapshot['attempt']}/{MAX_RETRIES} 次 "
        f"| 输入处理≈{input_speed:7.1f}字符/s | 当前输出={current_chars_per_second:6.1f}字符/s "
        f"| 已收={snapshot['output_chars']}字符 | 按 S 跳过本题"
    )
    return line, snapshot["output_chars"]


def _drain_progress(progress_queue: Any, state: dict[str, Any]) -> None:
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            return
        if event[0] == "update":
            state.update(event[1])
        elif event[0] == "output":
            count, timestamp = event[1], event[2]
            if count <= 0:
                continue
            if state["first_output_at"] is None:
                state["first_output_at"] = timestamp
            state["last_output_at"] = timestamp
            state["output_chars"] += count


def _terminate_api_process(process: Any) -> None:
    """Hard-stop the request and confirm it is gone before starting another one."""
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=2)


def _clear_status_line(last_line_length: int) -> None:
    sys.stdout.write("\r" + (" " * last_line_length) + "\r")
    sys.stdout.flush()


def _stdout_supports_in_place_status() -> bool:
    """Return True only for a real console that interprets carriage returns."""
    if os.name == "nt":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
        except (AttributeError, OSError):
            return False
    return bool(sys.stdout.isatty())


def api_chat(prompt: str, question_id: str, method: str, order: str,
             max_call_seconds: float, first_output_timeout_seconds: float) -> str:
    """Run one streaming API call in a killable process with a one-line monitor."""
    started = time.monotonic()
    state: dict[str, Any] = {
        "question_id": question_id, "method": method, "order": order,
        "started_at": started, "attempt_started_at": started,
        "input_chars": _estimate_input_chars(prompt),
        "first_output_at": None, "last_output_at": None, "output_chars": 0,
        "attempt": 1, "phase": "connecting", "retry_error": "",
    }
    api_config = {
        "api_key": JUDGE_API_KEY, "model": JUDGE_MODEL_NAME,
        "base_url": JUDGE_BASE_URL, "temperature": TEMPERATURE, "seed": SEED,
        "max_retries": MAX_RETRIES, "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "retry_base_seconds": RETRY_BASE_SECONDS,
    }
    context = mp.get_context("spawn")
    results = context.Queue(maxsize=1)
    progress = context.Queue()
    worker = context.Process(
        target=_api_chat_process, args=(prompt, api_config, results, progress), daemon=True)
    worker.start()

    live_status = _stdout_supports_in_place_status()
    if not live_status:
        timeout_text = (f"{first_output_timeout_seconds:g}s" if first_output_timeout_seconds > 0
                        else "关闭")
        print(
            f"API调用 | 题目 {question_id} | {method}/{order} | 等待结果 "
            f"| 首输出超时={timeout_text} | 按 S 跳过本题",
            flush=True,
        )
    report_interval = STATUS_REFRESH_SECONDS
    last_report = started - report_interval
    previous_time = started
    previous_output_chars = 0
    last_line_length = 0
    while True:
        now = time.monotonic()
        _drain_progress(progress, state)
        if _skip_key_pressed():
            _terminate_api_process(worker)
            if live_status:
                _clear_status_line(last_line_length)
            print(f"[SKIP] 用户跳过题目 {question_id}（当前调用 {method}/{order}）。", flush=True)
            raise SkipQuestion("user pressed S")
        if (first_output_timeout_seconds > 0 and state["first_output_at"] is None
                and state["phase"] in {"connecting", "waiting_first_output"}
                and now - state["attempt_started_at"] >= first_output_timeout_seconds):
            _terminate_api_process(worker)
            if live_status:
                _clear_status_line(last_line_length)
            print(
                f"[TIMEOUT] 题目 {question_id} {method}/{order} 在 "
                f"{first_output_timeout_seconds:g}s 内无首输出，已终止该调用并继续。",
                flush=True,
            )
            raise TimeoutError("no first API output before configured timeout")
        if max_call_seconds > 0 and now - started >= max_call_seconds:
            _terminate_api_process(worker)
            if live_status:
                _clear_status_line(last_line_length)
            print(f"[SKIP] 题目 {question_id} 单次调用超过 {max_call_seconds:g}s，自动跳过。", flush=True)
            raise SkipQuestion("maximum call time exceeded")
        if live_status and now - last_report >= report_interval:
            line, current_count = _progress_line(state, now, previous_output_chars, previous_time)
            sys.stdout.write("\r" + line.ljust(last_line_length))
            sys.stdout.flush()
            last_line_length = max(last_line_length, len(line))
            previous_output_chars = current_count
            previous_time = now
            last_report = now
        try:
            kind, value = results.get_nowait()
            worker.join(timeout=2)
            if live_status:
                _clear_status_line(last_line_length)
            if kind == "result":
                return str(value)
            raise RuntimeError(str(value))
        except queue.Empty:
            if not worker.is_alive():
                # Queue feeder may flush a few milliseconds after process exit.
                try:
                    kind, value = results.get(timeout=0.5)
                except queue.Empty:
                    if live_status:
                        _clear_status_line(last_line_length)
                    raise RuntimeError("API worker ended without returning a result")
                if live_status:
                    _clear_status_line(last_line_length)
                if kind == "result":
                    return str(value)
                raise RuntimeError(str(value))
            time.sleep(0.1)


def mock_chat(prompt: str) -> str:
    """Deterministic plumbing test; not a substitute for an LLM Judge."""
    digest = sum(ord(ch) for ch in prompt) % 17
    winner = "tie" if digest == 0 else ("A" if digest % 3 else "B")
    key = "final_winner" if "final_winner" in prompt else "winner"
    return json.dumps({"reason": "Deterministic mock output for pipeline testing only.", key: winner})


def parse_winner(raw: str, method: str) -> str:
    key = "final_winner" if method == "reason" else "winner"
    text = raw.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")
        obj = json.loads(match.group(0))
    value = str(obj.get(key, "")).strip()
    if value.lower() == "tie":
        return "tie"
    value = value.upper()
    if value not in {"A", "B"}:
        raise ValueError(f"Invalid {key}: {value!r}")
    return value


def map_to_candidate(winner: str, order: str) -> str:
    if winner == "tie":
        return "tie"
    if order == "forward":
        return "candidate_1" if winner == "A" else "candidate_2"
    return "candidate_2" if winner == "A" else "candidate_1"


def run_judging(questions: list[dict[str, Any]], answers: list[dict[str, Any]],
                mock: bool, limit: int | None, skip_question_ids: set[str],
                max_call_seconds: float, first_output_timeout_seconds: float,
                answer_version: str) -> None:
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    if OUTPUTS_PATH.exists():
        for row in load_jsonl(OUTPUTS_PATH):
            existing[(str(row["question_id"]), row["method"], row["order"])] = row
    q_by_id = {str(x["question_id"]): x for x in questions}
    selected = answers[:limit] if limit else answers
    total = sum(str(item["question_id"]) not in skip_question_ids for item in selected) * 4
    completed = 0
    for answer in selected:
        qid = str(answer["question_id"])
        if qid in skip_question_ids:
            print(f"[SKIP] 预设跳过题目 {qid}。", flush=True)
            continue
        skip_current_question = False
        for method in ("baseline", "reason"):
            for order in ("forward", "reverse"):
                completed += 1
                key = (qid, method, order)
                # Never mistake a prior mock record for a formal API result (or vice versa).
                if (key in existing and existing[key].get("parse_ok")
                        and bool(existing[key].get("is_mock")) == mock):
                    continue
                prompt = build_prompt(q_by_id[qid], answer, order, method)
                prompt_row = {"question_id": answer["question_id"], "method": method,
                              "order": order, "system_prompt": SYSTEM_PROMPT,
                              "user_prompt": prompt, "answer_version": answer_version}
                append_jsonl(PROMPTS_PATH, prompt_row)
                try:
                    raw = (mock_chat(prompt) if mock else
                           api_chat(prompt, qid, method, order, max_call_seconds,
                                    first_output_timeout_seconds))
                    winner = parse_winner(raw, method)
                    row = {"question_id": answer["question_id"], "method": method,
                           "order": order, "raw_output": raw, "parsed_winner": winner,
                           "parse_ok": True, "is_mock": mock,
                           "answer_version": answer_version}
                except SkipQuestion as exc:
                    row = {"question_id": answer["question_id"], "method": method,
                           "order": order, "raw_output": "", "parsed_winner": None,
                           "parse_ok": False, "is_mock": mock, "skipped": True,
                           "error": str(exc), "answer_version": answer_version}
                    skip_current_question = True
                except Exception as exc:
                    row = {"question_id": answer["question_id"], "method": method,
                           "order": order, "raw_output": "", "parsed_winner": None,
                           "parse_ok": False, "is_mock": mock, "error": repr(exc),
                           "answer_version": answer_version}
                    append_jsonl(FAILURES_PATH, row)
                append_jsonl(OUTPUTS_PATH, row)
                existing[key] = row
                print(f"[{completed}/{total}] {qid} {method} {order}: {row['parsed_winner']}", flush=True)
                if skip_current_question:
                    break
            if skip_current_question:
                break


def rate(values: list[str], target: str) -> float:
    return sum(v == target for v in values) / len(values) if values else float("nan")


def calculate_results(questions: list[dict[str, Any]], truth: list[dict[str, Any]],
                      limit: int | None, answer_version: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truth_by_id = {str(x["question_id"]): x for x in truth}
    valid_ids = [str(x["question_id"]) for x in (questions[:limit] if limit else questions)]
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in load_jsonl(OUTPUTS_PATH):
        if row.get("parse_ok"):
            latest[(str(row["question_id"]), row["method"], row["order"])] = row
    mapped: list[dict[str, Any]] = []
    for qid in valid_ids:
        t = truth_by_id[qid]
        out: dict[str, Any] = {"question_id": qid, "answer_version": f"v{answer_version}"}
        complete = True
        for method in ("baseline", "reason"):
            for order in ("forward", "reverse"):
                row = latest.get((qid, method, order))
                if not row:
                    complete = False
                    continue
                candidate = map_to_candidate(row["parsed_winner"], order)
                label = ("tie" if candidate == "tie" else
                         "strong" if candidate == t["strong_candidate"] else "weak")
                out[f"{method}_{order}_winner"] = label
            if complete:
                f, r = out[f"{method}_forward_winner"], out[f"{method}_reverse_winner"]
                out[f"{method}_consistent"] = f == r
                out[f"{method}_merged_winner"] = f if f == r else "tie"
                out[f"{method}_forced_tie"] = f != r
        if complete:
            mapped.append(out)
    if not mapped:
        raise RuntimeError("No complete question results are available")

    def summarize_pair(method: str) -> dict[str, float]:
        f = [x[f"{method}_forward_winner"] for x in mapped]
        r = [x[f"{method}_reverse_winner"] for x in mapped]
        all_values = f + r
        consistency = sum(x[f"{method}_consistent"] for x in mapped) / len(mapped)
        return {"consistency": consistency, "flip_rate": 1 - consistency,
                "accuracy": rate(all_values, "strong"), "strong_win_rate": rate(all_values, "strong"),
                "weak_win_rate": rate(all_values, "weak"), "tie_rate": rate(all_values, "tie")}

    def one_order(method: str, order: str) -> dict[str, Any]:
        values = [x[f"{method}_{order}_winner"] for x in mapped]
        return {"consistency": "", "flip_rate": "", "accuracy": rate(values, "strong"),
                "strong_win_rate": rate(values, "strong"), "weak_win_rate": rate(values, "weak"),
                "tie_rate": rate(values, "tie"), "forced_tie_rate": ""}

    def merged(method: str) -> dict[str, Any]:
        values = [x[f"{method}_merged_winner"] for x in mapped]
        forced = sum(x[f"{method}_forced_tie"] for x in mapped) / len(mapped)
        return {"consistency": "", "flip_rate": "", "accuracy": rate(values, "strong"),
                "strong_win_rate": rate(values, "strong"), "weak_win_rate": rate(values, "weak"),
                "tie_rate": rate(values, "tie"), "forced_tie_rate": forced}

    rows: list[dict[str, Any]] = []
    for condition, data in [
        ("Baseline Forward", one_order("baseline", "forward")),
        ("Baseline Reverse", one_order("baseline", "reverse")),
        ("Baseline Overall", {**summarize_pair("baseline"), "forced_tie_rate": ""}),
        ("Swap-then-Merge", merged("baseline")),
        ("Reason-then-Judge Forward", one_order("reason", "forward")),
        ("Reason-then-Judge Reverse", one_order("reason", "reverse")),
        ("Reason-then-Judge Overall", {**summarize_pair("reason"), "forced_tie_rate": ""}),
        ("Reason-then-Judge + Swap-then-Merge", merged("reason")),
    ]:
        rows.append({"answer_version": f"v{answer_version}",
                     "experiment_condition": condition, "n_questions": len(mapped), **data})
    return mapped, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "—" if value == "" else f"{value:.3f}" if isinstance(value, float) else str(value)


def write_report(metrics: list[dict[str, Any]], mock: bool, answer_version: str) -> None:
    headers = ["experiment_condition", "consistency", "flip_rate", "accuracy",
               "strong_win_rate", "weak_win_rate", "tie_rate", "forced_tie_rate"]
    title = ("# LLM Judge 长度偏见实验报告" if answer_version == "3"
             else "# LLM Judge 表层说服偏见实验报告" if answer_version == "4"
             else "# LLM Judge 位置偏见实验报告")
    lines = [title, "",
             f"> 运行模式：{'MOCK（仅验证流程，不可作为实验结论）' if mock else '真实 Judge API'}", "",
             f"> 候选回答版本：V{answer_version}", "",
             "## 指标汇总", "", "| 条件 | Consistency | Flip Rate | Accuracy | Strong Win | Weak Win | Tie | Forced Tie |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append("| " + " | ".join([row[headers[0]]] + [fmt(row[h]) for h in headers[1:]]) + " |")
    b = next(x for x in metrics if x["experiment_condition"] == "Baseline Overall")
    r = next(x for x in metrics if x["experiment_condition"] == "Reason-then-Judge Overall")
    bm = next(x for x in metrics if x["experiment_condition"] == "Swap-then-Merge")
    rm = next(x for x in metrics if x["experiment_condition"] == "Reason-then-Judge + Swap-then-Merge")
    lines += ["", "## 解释模板", "",
              f"- 基线交换顺序后的翻转率为 {fmt(b['flip_rate'])}。明显高于 0 表明 Judge 对呈现位置敏感；应同时结合正序与逆序准确率差异判断偏好方向。",
              f"- Reason-then-Judge 的翻转率为 {fmt(r['flip_rate'])}。若低于基线且准确率不下降，则该提示干预有效。",
              f"- Swap-then-Merge 后强回答胜率为 {fmt(bm['strong_win_rate'])}，强制平局率为 {fmt(bm['forced_tie_rate'])}。它以更多平局换取对冲突结论的保守处理。",
              f"- 两种方法结合后的强回答胜率为 {fmt(rm['strong_win_rate'])}，强制平局率为 {fmt(rm['forced_tie_rate'])}。应与单独干预比较，而不能只看翻转率。",
              "- Accuracy 按全部有效判决中选择 strong 的比例计算；tie 不计正确。题面公式中的第二项应为加号，而非减号。", ""]
    if answer_version == "3":
        b_weak = fmt(b["weak_win_rate"])
        r_weak = fmt(r["weak_win_rate"])
        lines += [
            "## 长度偏见（V3 专用）", "",
            "V3 中弱回答是 V2 弱回答的注水扩写版：内容（含错误）不变，长度≥2 倍。"
            "判断长度偏见的核心对照是 V3 与 V2 的弱回答胜率之差：",
            "",
            f"- 基线弱回答（注水版）胜率为 {b_weak}。若明显高于 V2 同一条件的弱回答胜率，"
            "说明 Judge 因冗长而高估弱回答，即存在长度偏见。",
            f"- Reason-then-Judge 的弱回答胜率为 {r_weak}。若该干预能把胜率压回 V2 水平，"
            "说明先推理后判决有助于抵抗长度偏见。",
             "- 也可比较 V3 与 V2 的 Accuracy 差值；Accuracy 下降越多，长度偏见越严重。", ""]
    if answer_version == "4":
        b_weak = fmt(b["weak_win_rate"])
        r_weak = fmt(r["weak_win_rate"])
        lines += [
            "## 表层说服偏见（V4 专用）", "",
            "V4 完整保留 V2 弱回答原文，仅添加固定的权威、群体共识、常识化、精致标题与同情表达。"
            "核心对照是 V4 与 V2 在同一 Judge 和同一条件下的弱回答胜率差：", "",
            f"- 基线弱回答（表层增强版）胜率为 {b_weak}。若明显高于 V2 的基线弱回答胜率，"
            "说明 Judge 可能被不增加实质内容的权威/共识/风格/同情线索误导。",
            f"- Reason-then-Judge 的弱回答胜率为 {r_weak}。若该干预能把胜率压回 V2 水平，"
            "说明显式比较内容质量可能有助于抵抗表层说服偏见。",
            "- V4 将多种表层线索组合为一个处理条件，因此可检验总体效应，但不能仅凭本版本"
            "把总体效应唯一归因于某一种线索；若需区分三类偏见，应进一步建立单因素消融版本。", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Test the pipeline without API calls")
    parser.add_argument(
        "--test-api", action="store_true",
        help="Test the configured Judge API and exit without running the experiment")
    parser.add_argument(
        "--skip-api-check", action="store_true",
        help="Skip the automatic API availability check before a real experiment")
    parser.add_argument(
        "--api-test-timeout", type=float, default=DEFAULT_API_TEST_TIMEOUT_SECONDS,
        help="Timeout in seconds for the startup API check (default: 15)")
    parser.add_argument("--limit", type=int, help="Run only the first N questions (smoke tests)")
    parser.add_argument("--fresh", action="store_true", help="Start new prompt/output/log files")
    parser.add_argument(
        "--answer-version", choices=("1", "2", "3", "4"),
        help="Candidate answer set: 1=larger quality gap, 2=close quality, "
             "3=verbosity bias (weak answer padded to >=2x length), "
             "4=surface-persuasion bias (fixed authority/consensus/style/compassion cues)")
    parser.add_argument(
        "--answer-versions", metavar="LIST",
        help="Run multiple answer sets, e.g. 2,3,4 (results use VxRy folders)")
    parser.add_argument(
        "--rounds", type=int,
        help="Number of independent runs for every selected answer version")
    parser.add_argument(
        "--skip-question", action="append", default=[], metavar="ID[,ID]",
        help="Skip one or more complete questions, e.g. --skip-question 90,105")
    parser.add_argument(
        "--max-call-seconds", type=float, default=DEFAULT_MAX_CALL_SECONDS,
        help="Automatically skip the current question when one API call exceeds this time; 0 disables")
    parser.add_argument(
        "--first-output-timeout", type=float,
        default=DEFAULT_FIRST_OUTPUT_TIMEOUT_SECONDS,
        help="Fail one judgment and continue if the API returns nothing within this many seconds; 0 disables")
    args = parser.parse_args()
    if args.mock and args.test_api:
        parser.error("--mock and --test-api cannot be used together")
    if args.api_test_timeout <= 0:
        parser.error("--api-test-timeout must be greater than zero")
    if args.test_api:
        print(f"Testing Judge API | model={JUDGE_MODEL_NAME} | endpoint={_chat_completions_url(JUDGE_BASE_URL)}")
        try:
            result = test_api_connection(args.api_test_timeout)
        except RuntimeError as exc:
            parser.exit(1, f"API test failed: {exc}\n")
        print(f"API test passed | model={result['model']} | latency={result['elapsed_seconds']:.2f}s")
        return
    answer_versions, rounds, named_runs = choose_experiment_plan(args, parser)
    if args.limit is not None and not 1 <= args.limit <= 80:
        parser.error("--limit must be in [1, 80]")
    if args.max_call_seconds < 0:
        parser.error("--max-call-seconds must be nonnegative")
    if args.first_output_timeout < 0:
        parser.error("--first-output-timeout must be nonnegative")
    skip_question_ids = {
        item.strip()
        for group in args.skip_question
        for item in group.split(",")
        if item.strip()
    }
    questions = load_jsonl(QUESTIONS_PATH)
    truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    answers_by_version: dict[str, list[dict[str, Any]]] = {}
    for answer_version in answer_versions:
        answer_path = ROOT / f"answers_v{answer_version}.jsonl"
        answers = load_jsonl(answer_path)
        validate_inputs(questions, answers, truth)
        answers_by_version[answer_version] = answers
    known_ids = {str(q["question_id"]) for q in questions}
    unknown_skip_ids = skip_question_ids - known_ids
    if unknown_skip_ids:
        parser.error(f"unknown --skip-question IDs: {sorted(unknown_skip_ids)}")
    if not args.mock and not args.skip_api_check:
        print(f"Checking Judge API before experiment | model={JUDGE_MODEL_NAME}")
        try:
            result = test_api_connection(args.api_test_timeout)
        except RuntimeError as exc:
            parser.exit(
                1,
                f"API check failed; no experiment files were changed: {exc}\n"
                "Fix the API configuration, run --test-api for diagnostics, or use "
                "--skip-api-check to bypass this check.\n",
            )
        print(f"API check passed | model={result['model']} | latency={result['elapsed_seconds']:.2f}s")
    total_runs = len(answer_versions) * rounds
    print(
        f"Experiment plan: versions={','.join('V' + v for v in answer_versions)} | "
        f"rounds={rounds} | total runs={total_runs}"
    )
    run_index = 0
    for answer_version in answer_versions:
        for round_number in range(1, rounds + 1):
            run_index += 1
            run_name = f"V{answer_version}R{round_number}" if named_runs else None
            configure_answer_version(answer_version, run_name)
            display_name = run_name or f"V{answer_version} (legacy output paths)"
            print(f"\n=== [{run_index}/{total_runs}] Starting {display_name} ===", flush=True)
            if args.fresh:
                for path in (PROMPTS_PATH, OUTPUTS_PATH, FAILURES_PATH):
                    path.unlink(missing_ok=True)
            random.seed(SEED)
            run_judging(
                questions, answers_by_version[answer_version], args.mock, args.limit,
                skip_question_ids, args.max_call_seconds, args.first_output_timeout,
                answer_version,
            )
            mapped, metrics = calculate_results(questions, truth, args.limit, answer_version)
            write_csv(MAPPED_PATH, mapped)
            write_csv(METRICS_PATH, metrics)
            write_report(metrics, args.mock, answer_version)
            print(
                f"Completed {display_name}: {len(mapped)} complete questions. "
                f"Report: {REPORT_PATH}",
                flush=True,
            )
    print(f"\nDone: {total_runs} experiment run(s) completed.")


if __name__ == "__main__":
    main()
