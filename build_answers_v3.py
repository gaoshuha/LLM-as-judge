#!/usr/bin/env python3
"""Build the verbosity-bias V3 answer set from the close-quality V2 set.

V3 keeps candidate_1 (the strong answer) byte-for-byte identical to V2.
candidate_2 (the weak answer) is an LLM-rewritten "padded" version of the V2
weak answer: the content is preserved (including any mistakes), but the wording
is expanded to at least 2x the original character length. This isolates length
as the manipulated factor for the verbosity-bias experiment.

The script calls the API only once per turn and saves progress incrementally;
re-running resumes from answers_v3_progress.jsonl. Use --mock to test the
pipeline without API calls (mechanical repetition, not real padding).

Outputs:
  answers_v3.jsonl        - same schema as answers_v2.jsonl
  v3_length_report.csv    - per-turn original/expanded lengths and ratio
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# =========================================================
# Step 0: 扩写模型的 API 配置（与 Judge 相互独立，可换成任何兼容服务）
# 优先读取环境变量 EXPAND_API_KEY，未设置时使用占位符。
EXPAND_API_KEY = os.environ.get("EXPAND_API_KEY", "PASTE_YOUR_EXPAND_API_KEY_HERE")
EXPAND_MODEL_NAME = "deepseek-v4-flash"
EXPAND_BASE_URL = "https://api.deepseek.com/v1"

TEMPERATURE = 0.3
SEED = 0
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 180
MAX_EXPAND_ATTEMPTS = 4        # per turn, when length/number checks fail
MIN_LENGTH_RATIO = 2.0         # hard requirement: expanded >= 2x original chars
AIM_RATIO = 2.6                # what the prompt asks for, to leave headroom
DEFAULT_WORKERS = 4

# Per-turn form constraints discovered during manual review of the padded set.
# Key: (question_id, turn_index). The hint is appended to the expansion prompt.
FORM_HINTS: dict[tuple[int, int], str] = {
    (99, 0): (
        "Additional form constraint: the original is a short RHYMING POEM of "
        "fewer than 10 lines, and the user explicitly required that form. Your "
        "rewrite MUST remain a rhyming poem with at most 9 lines, keeping the "
        "same rhyme scheme and the same mathematical statements. Make each line "
        "longer and more elaborate instead of converting the poem to prose."
    ),
}

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "question.jsonl"
SOURCE_PATH = ROOT / "answers_v2.jsonl"
PROGRESS_PATH = ROOT / "answers_v3_progress.jsonl"
TARGET_PATH = ROOT / "answers_v3.jsonl"
LENGTH_REPORT_PATH = ROOT / "v3_length_report.csv"

EXPAND_SYSTEM_PROMPT = (
    "You are a rewriting tool. You make text longer without changing its "
    "meaning, facts, or quality level. You never add new information and "
    "never fix mistakes. You output only the rewritten text."
)

EXPAND_INSTRUCTION = """Rewrite the answer below to be MUCH more verbose while preserving its exact content.

Strict rules:
- Do NOT add new facts, claims, examples, steps, or corrections.
- Do NOT remove or alter any fact, number, name, or conclusion. If the original contains a mistake, keep the mistake exactly as is.
- If the original contains code, keep every code block BYTE-FOR-BYTE identical; expand only the surrounding prose.
- Expand only by saying the same things in more words: redundant transitions, restated framing, spelled-out implications that are already present, verbose filler phrasing, polite hedging.
- The original is {n_chars} characters long. Your rewrite MUST be at least {min_chars} characters long. Aim for about {aim_chars} characters.
- Output ONLY the rewritten answer. No preamble, no commentary, no wrapping quotes.{hint}

Original answer:
\"\"\"
{original}
\"\"\""""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def extract_numbers(text: str) -> list[str]:
    """Numbers are the most checkable carrier of factual content."""
    return re.findall(r"\d+(?:\.\d+)?", text)


def missing_numbers(original: str, expanded: str) -> list[str]:
    remaining = extract_numbers(expanded)
    missing: list[str] = []
    for num in extract_numbers(original):
        if num in remaining:
            remaining.remove(num)
        else:
            missing.append(num)
    return missing


def check_expansion(original: str, expanded: str) -> tuple[bool, str]:
    """Return (ok, reason). Hard checks: >=2x length and no lost numbers."""
    expanded = expanded.strip()
    if not expanded or expanded == original.strip():
        return False, "empty or unchanged"
    if len(expanded) < MIN_LENGTH_RATIO * len(original.strip()):
        return False, (f"too short ({len(expanded)} < {MIN_LENGTH_RATIO}*"
                       f"{len(original.strip())})")
    lost = missing_numbers(original, expanded)
    if lost:
        return False, f"lost numbers: {lost[:5]}"
    return True, "ok"


def api_expand(original: str, aim_ratio: float, hint: str = "") -> str:
    """One non-streaming chat completion asking for a padded rewrite."""
    url = EXPAND_BASE_URL.rstrip("/") + "/chat/completions"
    n_chars = len(original.strip())
    prompt = EXPAND_INSTRUCTION.format(
        n_chars=n_chars,
        min_chars=int(MIN_LENGTH_RATIO * n_chars) + 1,
        aim_chars=int(aim_ratio * n_chars),
        original=original,
        hint=("\n\n" + hint) if hint else "",
    )
    payload = {
        "model": EXPAND_MODEL_NAME,
        "messages": [
            {"role": "system", "content": EXPAND_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "seed": SEED,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {EXPAND_API_KEY}",
                 "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request,
                                        timeout=REQUEST_TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"]).strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                KeyError, IndexError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(RETRY_BASE_SECONDS * (2 ** attempt))
    raise RuntimeError(f"expand API failed after {MAX_RETRIES} attempts: {last_error}")


def mock_expand(original: str, aim_ratio: float) -> str:
    """Deterministic mechanical padding for pipeline testing only."""
    text = original.strip()
    fillers = ["In other words, ", "To restate the same point more fully, ",
               "Putting it once again in a slightly more elaborate way, ",
               "As has already been said above, "]
    parts = [text]
    index = 0
    while len("\n\n".join(parts)) < aim_ratio * len(text):
        parts.append(fillers[index % len(fillers)] + text)
        index += 1
    return "\n\n".join(parts)


def expand_turn(original: str, mock: bool, hint: str = "") -> dict[str, Any]:
    """Expand one turn, retrying with a higher target when checks fail."""
    best: dict[str, Any] | None = None
    last_reason = "no attempt"
    for attempt in range(MAX_EXPAND_ATTEMPTS):
        aim = AIM_RATIO + 0.5 * attempt
        expanded = (mock_expand(original, aim) if mock
                    else api_expand(original, aim, hint))
        ok, reason = check_expansion(original, expanded)
        candidate = {"expanded": expanded, "attempts": attempt + 1,
                     "length_ok": len(expanded) >= MIN_LENGTH_RATIO * len(original.strip()),
                     "numbers_ok": not missing_numbers(original, expanded)}
        if best is None or len(expanded) > len(best["expanded"]):
            best = candidate
        if ok:
            return candidate
        last_reason = reason
    # Keep the longest attempt rather than dropping the sample; it is flagged
    # in the length report so it can be reviewed or regenerated.
    print(f"[WARN] a turn kept a padded answer failing checks: {last_reason}",
          flush=True)
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="Mechanical padding, no API calls (pipeline test only)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Concurrent API requests (default 4)")
    parser.add_argument("--fresh", action="store_true",
                        help="Discard prior expansion progress and start over")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if EXPAND_API_KEY == "PASTE_YOUR_EXPAND_API_KEY_HERE" and not args.mock:
        parser.error("Set EXPAND_API_KEY at the top of this file, or run with --mock.")

    questions = {str(q["question_id"]): q for q in load_jsonl(QUESTIONS_PATH)}
    source = load_jsonl(SOURCE_PATH)
    if args.fresh:
        PROGRESS_PATH.unlink(missing_ok=True)

    done: dict[tuple[str, int], dict[str, Any]] = {}
    if PROGRESS_PATH.exists():
        for row in load_jsonl(PROGRESS_PATH):
            if row.get("is_mock") == args.mock:
                done[(str(row["question_id"]), int(row["turn_index"]))] = row

    tasks: list[tuple[str, int, str]] = []
    for row in source:
        qid = str(row["question_id"])
        for turn_index, original in enumerate(row["candidate_2"]):
            if (qid, turn_index) not in done:
                tasks.append((qid, turn_index, original))
    print(f"V3 padding: {len(done)} turns already done, {len(tasks)} to expand "
          f"({'mock' if args.mock else EXPAND_MODEL_NAME}).", flush=True)

    lock = threading.Lock()
    finished = 0

    def work(qid: str, turn_index: int, original: str) -> dict[str, Any]:
        hint = FORM_HINTS.get((int(qid), turn_index), "")
        result = expand_turn(original, args.mock, hint)
        return {"question_id": int(qid), "turn_index": turn_index,
                "category": questions[qid]["category"],
                "original": original, "expanded": result["expanded"],
                "attempts": result["attempts"], "length_ok": result["length_ok"],
                "numbers_ok": result["numbers_ok"], "is_mock": args.mock}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, qid, ti, orig): (qid, ti)
                   for qid, ti, orig in tasks}
        for future in as_completed(futures):
            qid, ti = futures[future]
            row = future.result()
            with lock:
                with PROGRESS_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[(qid, ti)] = row
                finished += 1
                ratio = len(row["expanded"]) / max(len(row["original"].strip()), 1)
                print(f"[{len(done)}/{len(done) + len(tasks) - finished}] "
                      f"q{qid} turn{ti + 1}: {len(row['original'])} -> "
                      f"{len(row['expanded'])} chars (x{ratio:.2f}, "
                      f"attempts={row['attempts']})", flush=True)

    # Assemble the final V3 set in source order; candidate_1 is untouched.
    output = []
    report_rows = []
    for row in source:
        qid = str(row["question_id"])
        padded = []
        for turn_index, original in enumerate(row["candidate_2"]):
            rec = done[(qid, turn_index)]
            padded.append(rec["expanded"])
            ratio = len(rec["expanded"]) / max(len(original.strip()), 1)
            report_rows.append({
                "question_id": qid, "turn": turn_index + 1,
                "category": questions[qid]["category"],
                "original_chars": len(original.strip()),
                "expanded_chars": len(rec["expanded"]),
                "ratio": f"{ratio:.2f}", "length_ok": rec["length_ok"],
                "numbers_ok": rec["numbers_ok"], "attempts": rec["attempts"],
            })
        output.append({"question_id": row["question_id"],
                       "candidate_1": row["candidate_1"],
                       "candidate_2": padded})

    ratios = [float(r["ratio"]) for r in report_rows]
    short = [r for r in report_rows if not r["length_ok"]]
    lost = [r for r in report_rows if not r["numbers_ok"]]
    with TARGET_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for row in output:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with LENGTH_REPORT_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Wrote {TARGET_PATH.name}: {len(output)} questions, "
          f"{len(report_rows)} padded turns.", flush=True)
    print(f"Length ratio min/median/max: {min(ratios):.2f}/"
          f"{sorted(ratios)[len(ratios) // 2]:.2f}/{max(ratios):.2f}", flush=True)
    print(f"Turns below {MIN_LENGTH_RATIO}x: {len(short)}; "
          f"turns with lost numbers: {len(lost)}. "
          f"Details: {LENGTH_REPORT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
