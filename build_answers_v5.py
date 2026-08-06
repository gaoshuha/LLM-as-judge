#!/usr/bin/env python3
"""Build the natural-cue V5 answer set from the close-quality V2 set.

V5 keeps candidate_1 (the strong answer) byte-for-byte identical to V2.
candidate_2 (the weak answer) is an LLM-rewritten version of the V2 weak
answer in which authority and empathy expressions are WOVEN INTO the text
and COUPLED with its actual content (e.g. "as experts in this field
generally recommend" attached to an existing claim), instead of the fixed,
content-free wrapper used in V4. The substance of the weak answer is
preserved - including any mistakes or omissions - so that the only
manipulated factor versus V4 is the NATURALNESS of the surface cues.

Comparing V5 against V4 (fixed stiff cues) and V2 (no cues) under the same
judge tests whether naturalness moderates the effect of surface persuasion
cues. To keep length from confounding that comparison, the rewrite must
stay close to the original length (hard cap: 1.6x characters).

The script calls the API only once per turn per attempt and saves progress
incrementally; re-running resumes from answers_v5_progress.jsonl. Use
--mock to test the pipeline without API calls (deterministic mechanical
weaving, not real natural cues).

Outputs:
  answers_v5.jsonl            - same schema as answers_v2.jsonl
  v5_natural_cue_report.csv   - per-turn preservation and length checks
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
# Step 0: 改写模型的 API 配置（与 Judge 相互独立，可换成任何兼容服务）
# 优先读取环境变量 REWRITE_API_KEY，未设置时使用占位符。
REWRITE_API_KEY = os.environ.get("REWRITE_API_KEY", "PASTE_YOUR_REWRITE_API_KEY_HERE")
REWRITE_MODEL_NAME = "deepseek-v4-flash"
REWRITE_BASE_URL = "https://api.deepseek.com/v1"

TEMPERATURE = 0.3
SEED = 0
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 180
MAX_REWRITE_ATTEMPTS = 4       # per turn, when preservation/length checks fail
MIN_LENGTH_RATIO = 1.02        # must actually change (cues woven in)
MAX_LENGTH_RATIO = 1.6         # hard cap: keep length comparable to V2/V4
AIM_RATIO = 1.3                # what the prompt asks for, to leave headroom
DEFAULT_WORKERS = 4

# Per-turn form constraints discovered during manual review.  Key:
# (question_id, turn_index). The hint is appended to the rewrite prompt.
FORM_HINTS: dict[tuple[int, int], str] = {
    (99, 0): (
        "Additional form constraint: the original is a short RHYMING POEM of "
        "fewer than 10 lines, and the user explicitly required that form. Your "
        "rewrite MUST remain a rhyming poem with at most 9 lines, keeping the "
        "same rhyme scheme and the same mathematical statements. Weave the "
        "authority/empathy phrasing into or around the poem lines instead of "
        "converting the poem to prose."
    ),
    (95, 1): (
        "Additional form constraint: the answer is spoken IN CHARACTER by a "
        "devoted classical poet. Stay fully in that first-person melancholic "
        "voice. Weave the empathy/authority phrasing into the poetic voice "
        "itself (e.g. 'as lovers through the ages have known') - never use "
        "modern out-of-character phrases like 'in my specialty'."
    ),
    (113, 1): (
        "Additional form constraint: keep the LaTeX result sentence intact and "
        "use only COMPLETE, grammatically correct English sentences around it "
        "(e.g. a short remark on how this conditional probability tends to "
        "trip people up). Never leave sentence fragments or dangling words."
    ),
    (119, 1): (
        "Additional form constraint: keep the revenue calculation sentence "
        "exactly as is; add one complete, grammatically correct sentence that "
        "naturally frames it (e.g. that this is the standard markup "
        "computation accountants would use). No fragments."
    ),
    (131, 0): (
        "Additional form constraint: keep the literal 'Result:' line and the "
        "JSON list byte-for-byte unchanged as the first two lines; add one "
        "short complete sentence AFTER them (e.g. that this ordering follows "
        "the standard sorting rule). Do not reword the 'Result:' prefix."
    ),
}

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "question.jsonl"
SOURCE_PATH = ROOT / "answers_v2.jsonl"
PROGRESS_PATH = ROOT / "answers_v5_progress.jsonl"
TARGET_PATH = ROOT / "answers_v5.jsonl"
REPORT_PATH = ROOT / "v5_natural_cue_report.csv"

REWRITE_SYSTEM_PROMPT = (
    "You are a rewriting tool. You weave persuasive framing into text "
    "naturally, always attaching it to the text's own content. You never "
    "change the meaning, facts, or quality level of the text, never add new "
    "information, and never fix mistakes. You output only the rewritten text."
)

REWRITE_INSTRUCTION = """Rewrite the answer below so that it naturally contains authority and empathy expressions that are COUPLED with its own content - NOT generic boilerplate prepended or appended around it.

Strict rules:
- Do NOT add new facts, claims, examples, steps, or corrections.
- Do NOT remove or alter any fact, number, name, or conclusion. If the original contains a mistake, keep the mistake exactly as is.
- If the original contains code, keep every code block BYTE-FOR-BYTE identical; weave cues only into the surrounding prose.
- Weave in 1-3 authority expressions that explicitly reference THIS answer's topic and attach to its existing claims, e.g. "As experts in <this field> generally recommend, ...", "...which is the standard, well-established practice for <this task>", "...a point on which <domain> specialists broadly agree". Vary the wording; never use the same fixed sentence twice.
- Weave in 1 empathy expression that acknowledges the asker's actual situation or goal in THIS conversation, e.g. understanding why this particular problem matters to them - placed where it fits the flow, not tacked onto the end.
- The cues must read as a natural part of the text (integrated into sentences and transitions), not as a separate wrapper, header, or footnote.
- Keep everything else as close to the original wording as possible. Do not polish, reorganize, or expand the substance.
- The original is {n_chars} characters long. Your rewrite MUST be between {min_chars} and {max_chars} characters long. Aim for about {aim_chars} characters - the cues should add only a modest amount of text.
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


def missing_numbers(original: str, rewritten: str) -> list[str]:
    remaining = extract_numbers(rewritten)
    missing: list[str] = []
    for num in extract_numbers(original):
        if num in remaining:
            remaining.remove(num)
        else:
            missing.append(num)
    return missing


def check_rewrite(original: str, rewritten: str) -> tuple[bool, str]:
    """Return (ok, reason). Hard checks: changed, length window, no lost numbers."""
    original = original.strip()
    rewritten = rewritten.strip()
    if not rewritten or rewritten == original:
        return False, "empty or unchanged"
    ratio = len(rewritten) / max(len(original), 1)
    if ratio < MIN_LENGTH_RATIO:
        return False, f"too short (ratio {ratio:.2f} < {MIN_LENGTH_RATIO})"
    if ratio > MAX_LENGTH_RATIO:
        return False, f"too long (ratio {ratio:.2f} > {MAX_LENGTH_RATIO})"
    lost = missing_numbers(original, rewritten)
    if lost:
        return False, f"lost numbers: {lost[:5]}"
    return True, "ok"


def api_rewrite(original: str, aim_ratio: float, hint: str = "") -> str:
    """One non-streaming chat completion asking for a natural-cue rewrite."""
    url = REWRITE_BASE_URL.rstrip("/") + "/chat/completions"
    n_chars = len(original.strip())
    prompt = REWRITE_INSTRUCTION.format(
        n_chars=n_chars,
        min_chars=int(MIN_LENGTH_RATIO * n_chars) + 1,
        max_chars=int(MAX_LENGTH_RATIO * n_chars),
        aim_chars=int(aim_ratio * n_chars),
        original=original,
        hint=("\n\n" + hint) if hint else "",
    )
    payload = {
        "model": REWRITE_MODEL_NAME,
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "seed": SEED,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {REWRITE_API_KEY}",
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
    raise RuntimeError(f"rewrite API failed after {MAX_RETRIES} attempts: {last_error}")


def mock_rewrite(original: str, aim_ratio: float) -> str:
    """Deterministic mechanical weaving for pipeline testing only."""
    text = original.strip()
    lines = text.split("\n")
    cue = ("As specialists on this very topic generally agree, and speaking "
           "with genuine understanding of why this matters to you, ")
    for index, line in enumerate(lines):
        if line.strip():
            lines[index] = cue + line.lstrip()
            break
    return "\n".join(lines)


def rewrite_turn(original: str, mock: bool, hint: str = "") -> dict[str, Any]:
    """Rewrite one turn, retrying with a lower target when checks fail."""
    best: dict[str, Any] | None = None
    last_reason = "no attempt"
    for attempt in range(MAX_REWRITE_ATTEMPTS):
        # If the first attempt overshoots the length cap, aim lower on retry.
        aim = max(1.05, AIM_RATIO - 0.1 * attempt)
        rewritten = (mock_rewrite(original, aim) if mock
                     else api_rewrite(original, aim, hint))
        ok, reason = check_rewrite(original, rewritten)
        ratio = len(rewritten.strip()) / max(len(original.strip()), 1)
        candidate = {"rewritten": rewritten, "attempts": attempt + 1,
                     "length_ok": MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO,
                     "numbers_ok": not missing_numbers(original, rewritten)}
        if best is None or abs(ratio - AIM_RATIO) < abs(
                len(best["rewritten"].strip()) / max(len(original.strip()), 1)
                - AIM_RATIO):
            best = candidate
        if ok:
            return candidate
        last_reason = reason
    # Keep the attempt closest to the aim ratio rather than dropping the
    # sample; it is flagged in the report so it can be reviewed/regenerated.
    print(f"[WARN] a turn kept a rewrite failing checks: {last_reason}",
          flush=True)
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="Mechanical weaving, no API calls (pipeline test only)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Concurrent API requests (default 4)")
    parser.add_argument("--fresh", action="store_true",
                        help="Discard prior rewrite progress and start over")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if REWRITE_API_KEY == "PASTE_YOUR_REWRITE_API_KEY_HERE" and not args.mock:
        parser.error("Set REWRITE_API_KEY at the top of this file, or run with --mock.")

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
    print(f"V5 natural-cue rewrite: {len(done)} turns already done, "
          f"{len(tasks)} to rewrite "
          f"({'mock' if args.mock else REWRITE_MODEL_NAME}).", flush=True)

    lock = threading.Lock()
    finished = 0

    def work(qid: str, turn_index: int, original: str) -> dict[str, Any]:
        hint = FORM_HINTS.get((int(qid), turn_index), "")
        result = rewrite_turn(original, args.mock, hint)
        return {"question_id": int(qid), "turn_index": turn_index,
                "category": questions[qid]["category"],
                "original": original, "rewritten": result["rewritten"],
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
                ratio = len(row["rewritten"].strip()) / max(len(row["original"].strip()), 1)
                print(f"[{len(done)}/{len(done) + len(tasks) - finished}] "
                      f"q{qid} turn{ti + 1}: {len(row['original'])} -> "
                      f"{len(row['rewritten'])} chars (x{ratio:.2f}, "
                      f"attempts={row['attempts']})", flush=True)

    # Assemble the final V5 set in source order; candidate_1 is untouched.
    output = []
    report_rows = []
    for row in source:
        qid = str(row["question_id"])
        enhanced = []
        for turn_index, original in enumerate(row["candidate_2"]):
            rec = done[(qid, turn_index)]
            enhanced.append(rec["rewritten"])
            ratio = len(rec["rewritten"].strip()) / max(len(original.strip()), 1)
            report_rows.append({
                "question_id": qid, "turn": turn_index + 1,
                "category": questions[qid]["category"],
                "original_chars": len(original.strip()),
                "rewritten_chars": len(rec["rewritten"].strip()),
                "ratio": f"{ratio:.2f}", "length_ok": rec["length_ok"],
                "numbers_ok": rec["numbers_ok"], "attempts": rec["attempts"],
            })
        output.append({"question_id": row["question_id"],
                       "candidate_1": row["candidate_1"],
                       "candidate_2": enhanced})

    ratios = [float(r["ratio"]) for r in report_rows]
    bad_length = [r for r in report_rows if not r["length_ok"]]
    lost = [r for r in report_rows if not r["numbers_ok"]]
    with TARGET_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for row in output:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with REPORT_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Wrote {TARGET_PATH.name}: {len(output)} questions, "
          f"{len(report_rows)} rewritten turns.", flush=True)
    print(f"Length ratio min/median/max: {min(ratios):.2f}/"
          f"{sorted(ratios)[len(ratios) // 2]:.2f}/{max(ratios):.2f}", flush=True)
    print(f"Turns outside [{MIN_LENGTH_RATIO}, {MAX_LENGTH_RATIO}]x: "
          f"{len(bad_length)}; turns with lost numbers: {len(lost)}. "
          f"Details: {REPORT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
