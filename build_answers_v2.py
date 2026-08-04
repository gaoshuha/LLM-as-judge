#!/usr/bin/env python3
"""Build the close-quality V2 answer set from the independently authored V1 set.

V2 keeps candidate_1 unchanged. Candidate_2 is an editorially reduced version of
candidate_1: one nonessential explanation/detail is removed where possible. No
new factual claims or deliberate errors are introduced.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "answers_v1.jsonl"
QUESTIONS = ROOT / "question.jsonl"
TARGET = ROOT / "answers_v2.jsonl"

# Hand-reviewed cases where a purely mechanical deletion would remove the
# intended conclusion rather than a secondary detail.
MANUAL_CLOSE_OVERRIDES: dict[tuple[int, int], str] = {
    (95, 0): "Though I grow thin, I do not regret my devotion to you.",
    (108, 1): ("Replace **car** with **brake**; tyre, steering wheel, brake, "
               "and engine are all vehicle components."),
    (115, 1): ("If all passengers pay when boarding, \\(38+4+8=50\\) fares "
               "produce **$100** in revenue. The wording could clarify whether "
               "it instead means only the terminal cohort."),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def remove_code_explanation(text: str) -> str:
    """Keep executable code and its necessary lead-in, omit trailing commentary."""
    last_fence = text.rfind("```")
    if last_fence < 0:
        return text
    end = last_fence + 3
    trailing = text[end:].strip()
    return text[:end].strip() if trailing else text


def remove_one_line(text: str) -> str:
    lines = text.splitlines()
    content_indexes = [i for i, line in enumerate(lines) if line.strip()]
    if len(content_indexes) < 4:
        return text
    # Preserve first/last lines and remove one supporting line near the end.
    remove_at = content_indexes[-2]
    reduced = "\n".join(line for i, line in enumerate(lines) if i != remove_at).strip()
    # Do not delete a disproportionately large line.
    return reduced if len(reduced) >= 0.75 * len(text) else text


def split_sentences(text: str) -> list[str]:
    """Split prose while keeping closing quotation marks with their sentence."""
    stripped = text.strip()
    sentences: list[str] = []
    start = 0
    in_curly_quote = False
    in_straight_quote = False
    for index, char in enumerate(stripped):
        if char == "“":
            in_curly_quote = True
        elif char == "”":
            in_curly_quote = False
        elif char == '"' and (index == 0 or stripped[index - 1] != "\\"):
            in_straight_quote = not in_straight_quote
        next_is_boundary = index + 1 == len(stripped) or stripped[index + 1].isspace()
        punctuation_end = (char in ".!?。" and not in_curly_quote
                           and not in_straight_quote and next_is_boundary)
        quoted_end = (char in {'”', '"'} and index > 0 and stripped[index - 1] in ".!?。"
                      and not in_curly_quote and not in_straight_quote and next_is_boundary)
        if punctuation_end or quoted_end:
            sentences.append(stripped[start:index + 1].strip())
            start = index + 1
    remainder = stripped[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def remove_one_sentence(text: str, category: str) -> str:
    sentences = split_sentences(text)
    if len(sentences) >= 4:
        # Preserve framing and conclusion; omit one supporting qualification.
        del sentences[-2]
        reduced = " ".join(sentences)
        return reduced if len(reduced) >= 0.75 * len(text) else text
    if len(sentences) == 3:
        del sentences[1]
        reduced = " ".join(sentences)
        return reduced if len(reduced) >= 0.75 * len(text) else text
    if len(sentences) == 2:
        # In a two-sentence answer each sentence is often essential; retain both
        # instead of turning a worked answer into a bare conclusion.
        return text
    return text


def make_close_answer(text: str, category: str) -> str:
    if category == "extraction":
        # Exact extraction tasks have little legitimate room for factual
        # weakening. Preserve every extracted fact and add only a small format
        # redundancy; this is slightly less instruction-efficient, not false.
        return "Result:\n" + text
    if "```" in text:
        reduced = remove_code_explanation(text)
        return reduced if reduced != text and len(reduced) >= 0.70 * len(text) else text
    if text.count("\n") >= 3:
        reduced = remove_one_line(text)
        if reduced != text:
            return reduced
    return remove_one_sentence(text, category)


def force_minor_change(text: str, category: str) -> str:
    """Ensure each two-turn trajectory has at least one localized omission."""
    if "```" in text:
        reduced = remove_code_explanation(text)
        if reduced != text:
            return reduced
    sentences = split_sentences(text)
    if len(sentences) >= 2:
        # For worked problems retain the conclusion; elsewhere retain the main
        # response and omit the final qualification/detail.
        if category in {"math", "reasoning"} and len(sentences) == 2:
            return sentences[-1]
        remove_index = -2 if len(sentences) >= 3 else -1
        del sentences[remove_index]
        return " ".join(sentences)
    if text.count("\n") >= 2:
        lines = text.splitlines()
        for index in range(len(lines) - 1, -1, -1):
            if lines[index].strip():
                del lines[index]
                return "\n".join(lines).strip()
    if ";" in text:
        parts = text.split(";")
        if len(parts) >= 2:
            return ";".join(parts[:-1]).rstrip() + "."
    # A neutral lead-in is a tiny instruction-efficiency defect without adding
    # a false proposition.
    return "Briefly: " + text


def main() -> None:
    questions = {str(row["question_id"]): row for row in load_jsonl(QUESTIONS)}
    source = load_jsonl(SOURCE)
    output = []
    changed_turns = 0
    for row in source:
        qid = str(row["question_id"])
        category = questions[qid]["category"]
        close_candidate = []
        for strong_turn in row["candidate_1"]:
            reduced = make_close_answer(strong_turn, category)
            close_candidate.append(reduced)
            changed_turns += reduced != strong_turn
        if close_candidate == row["candidate_1"]:
            index = max(range(2), key=lambda i: len(row["candidate_1"][i]))
            close_candidate[index] = force_minor_change(
                row["candidate_1"][index], category)
            changed_turns += close_candidate[index] != row["candidate_1"][index]
        for turn_index in range(2):
            override = MANUAL_CLOSE_OVERRIDES.get((row["question_id"], turn_index))
            if override is not None:
                close_candidate[turn_index] = override
        output.append({
            "question_id": row["question_id"],
            "candidate_1": row["candidate_1"],
            "candidate_2": close_candidate,
        })
    with TARGET.open("w", encoding="utf-8", newline="\n") as fh:
        for row in output:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(output)} questions; reduced {changed_turns}/160 candidate turns")


if __name__ == "__main__":
    main()
