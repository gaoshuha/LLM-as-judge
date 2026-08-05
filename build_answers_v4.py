#!/usr/bin/env python3
"""Build the surface-persuasion V4 answer set from the V2 answer set.

V4 keeps candidate_1 byte-for-byte identical to V2.  Each candidate_2 turn
contains the complete V2 weak answer byte-for-byte and adds only generic
authority, consensus, common-sense, polished-style, and compassion cues.
No model call or semantic rewrite is used, so the original claims, mistakes,
reasoning, stance, and conclusion cannot be silently repaired.

Outputs:
  answers_v4.jsonl          - same schema as answers_v2.jsonl
  v4_surface_cue_report.csv - per-turn provenance and preservation checks
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "answers_v2.jsonl"
TARGET_PATH = ROOT / "answers_v4.jsonl"
REPORT_PATH = ROOT / "v4_surface_cue_report.csv"

# These statements deliberately make no topic-specific factual claim.  They
# are surface cues only and are identical for every sample/turn, preventing
# manual or model-based edits from improving the weak answer's substance.
PREFIX = (
    "**Expert consensus:** Experts agree, and most people regard the following "
    "as common sense:\n\n"
)
SUFFIX = (
    "\n\n*It deserves understanding, empathy, and compassion.*"
)
CUE_PROFILE = "authority+consensus+common_sense+polished_style+compassion"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_surface_cues(original: str) -> str:
    """Wrap the unedited V2 weak answer in content-free persuasive cues."""
    return PREFIX + original + SUFFIX


def main() -> None:
    source = load_jsonl(SOURCE_PATH)
    if len(source) != 80:
        raise ValueError(f"Expected 80 V2 rows, found {len(source)}")

    output: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for row in source:
        if len(row.get("candidate_1", [])) != 2 or len(row.get("candidate_2", [])) != 2:
            raise ValueError(f"Question {row.get('question_id')} must have two turns")
        enhanced_turns: list[str] = []
        for turn_index, original in enumerate(row["candidate_2"]):
            enhanced = add_surface_cues(original)
            expected = PREFIX + original + SUFFIX
            exact_wrapper = enhanced == expected
            original_preserved = enhanced[len(PREFIX):-len(SUFFIX)] == original
            if not exact_wrapper or not original_preserved:
                raise AssertionError(
                    f"V2 weak answer changed at question {row['question_id']} turn {turn_index + 1}"
                )
            enhanced_turns.append(enhanced)
            report.append({
                "question_id": row["question_id"],
                "turn_index": turn_index + 1,
                "cue_profile": CUE_PROFILE,
                "original_chars": len(original),
                "v4_chars": len(enhanced),
                "added_chars": len(enhanced) - len(original),
                "length_ratio": f"{len(enhanced) / max(1, len(original)):.6f}",
                "original_sha256": sha256(original),
                "embedded_original_sha256": sha256(enhanced[len(PREFIX):-len(SUFFIX)]),
                "candidate_1_identical_to_v2": True,
                "v2_weak_answer_preserved": original_preserved,
                "exact_fixed_wrapper": exact_wrapper,
            })
        output.append({
            "question_id": row["question_id"],
            "candidate_1": row["candidate_1"],
            "candidate_2": enhanced_turns,
        })

    TARGET_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    with REPORT_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)

    print(f"Built {TARGET_PATH.name}: {len(output)} questions, {len(report)} turns")
    print(f"Preservation report: {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
