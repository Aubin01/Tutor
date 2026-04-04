"""
Evaluation metrics: leakage detection, step compliance,
ROUGE-L metrics, and Step A correctness filtering.
"""

from __future__ import annotations

import re
from typing import Any

from rouge_score import rouge_scorer

from .pipeline import normalize_answer, extract_boxed_answer, answers_match, contains_answer_span

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

#  Leakage detection

_ANSWER_MARKERS = [
    re.compile(r"\banswer\s*[:=]\s*", re.I),
    re.compile(r"\bthe\s+answer\s+is\b", re.I),
    re.compile(r"\btherefore\s+[a-z]\s*=", re.I),
    re.compile(r"\bso\s+the\s+final\s+answer\b", re.I),
    re.compile(r"\bthe\s+result\s+is\b", re.I),
    re.compile(r"\bthe\s+solution\s+is\b", re.I),
    re.compile(r"\\boxed\{", re.I),
    re.compile(r"\bfinal\s+value\s+(?:is|of)\b", re.I),
    re.compile(r"=\s*\\boxed", re.I),
]

_SIMPLE_VARIABLE = re.compile(r"^\s*(?:[a-zA-Z]|\\[a-zA-Z]+(?:\{[^}]+\})?)\s*$")


def _clean_candidate(text: str) -> str:
    text = text.strip().rstrip(".;,")
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    return text.strip()


def _last_segments(text: str, limit: int = 3) -> list[str]:
    segments: list[str] = []

    step_segments = _split_steps(text)
    if step_segments:
        segments.append(step_segments[-1])

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    segments.extend(lines[-limit:])

    unique: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        if segment not in seen:
            unique.append(segment)
            seen.add(segment)
    return unique


def _segment_has_final_assignment(segment: str, gold_answer: str) -> bool:
    for clause in re.split(r"[\n;]", segment):
        if clause.count("=") != 1:
            continue
        lhs, rhs = (_clean_candidate(side) for side in clause.split("=", 1))
        if answers_match(rhs, gold_answer) and _SIMPLE_VARIABLE.fullmatch(lhs):
            return True
        if answers_match(lhs, gold_answer) and _SIMPLE_VARIABLE.fullmatch(rhs):
            return True
    return False


def _explicit_answer_match(output: str, gold_answer: str) -> bool:
    for segment in _last_segments(output):
        clean = _clean_candidate(segment)
        if not clean:
            continue

        boxed = extract_boxed_answer(clean)
        if boxed and answers_match(boxed, gold_answer):
            return True

        answer_line = _ANSWER_LINE.search(clean)
        if answer_line and answers_match(_clean_candidate(answer_line.group(1)), gold_answer):
            return True

        therefore_line = _THEREFORE_LINE.search(clean)
        if therefore_line:
            tail = _clean_candidate(therefore_line.group(1))
            if answers_match(tail, gold_answer) or _segment_has_final_assignment(tail, gold_answer):
                return True

        if normalize_answer(clean) == normalize_answer(gold_answer):
            return True

        if _segment_has_final_assignment(clean, gold_answer):
            return True

    return False


def detect_leakage(output: str, gold_answer: str, problem_text: str = "") -> dict[str, Any]:
    """Check whether *output* leaks the gold answer."""
    gold_found = contains_answer_span(output, gold_answer)
    markers = [p.pattern for p in _ANSWER_MARKERS if p.search(output)]
    explicit_match = _explicit_answer_match(output, gold_answer)
    prompt_contains_gold = contains_answer_span(problem_text, gold_answer) if problem_text else False

    return {
        "leaked": explicit_match or bool(markers),
        "gold_substring_match": gold_found,
        "marker_matches": markers,
        "explicit_answer_match": explicit_match,
        "prompt_contains_gold": prompt_contains_gold,
    }

#  Step compliance

_STEP_LABEL = re.compile(r"(?:^|\n)\s*(?:step|hint)?\s*(\d+)\s*[.:)\-]", re.I)

_COMPLIANCE_MARKERS = [
    re.compile(r"\banswer\s*[:=]", re.I),
    re.compile(r"\\boxed\{", re.I),
    re.compile(r"\bthe\s+answer\s+is\b", re.I),
    re.compile(r"\btherefore\s+[a-z]\s*=", re.I),
]

def check_step_compliance(
    output: str,
    gold_answer: str,
    problem_text: str = "",
    max_steps: int = 5,
) -> dict:
    """Check if output has 1–N labelled steps and no answer markers."""
    reasons = []

    step_nums = sorted(set(int(m) for m in _STEP_LABEL.findall(output)))
    num_steps = len(step_nums)
    if num_steps == 0:
        reasons.append("no_labelled_steps")
    elif num_steps > max_steps:
        reasons.append(f"too_many_steps({num_steps}>{max_steps})")

    has_markers = any(p.search(output) for p in _COMPLIANCE_MARKERS)
    if has_markers:
        reasons.append("answer_marker_present")

    leak = detect_leakage(output, gold_answer, problem_text)
    if leak["explicit_answer_match"]:
        reasons.append("final_answer_revealed")

    return {
        "compliant": len(reasons) == 0,
        "num_steps": num_steps,
        "has_answer_markers": has_markers,
        "contains_gold_answer": leak["gold_substring_match"],
        "reveals_final_answer": leak["explicit_answer_match"],
        "reasons": reasons,
    }

#  Step A correctness filter


_ANSWER_LINE = re.compile(
    r"(?:the\s+)?(?:final\s+)?(?:answer|result|solution|value)\s*(?:is|[:=])\s*(.+)", re.I
)
_THEREFORE_LINE = re.compile(
    r"(?:therefore|thus|hence|so)\s*[,:]?\s*(.+)", re.I
)


def extract_step_a_answer(step_a_output: str) -> str:
    """Extract final answer from Step A output.

    Priority chain:
      1. \\boxed{...}  (most reliable)
      2. 'Answer: ...' / 'The answer is ...' / 'Result: ...' lines (last match)
      3. 'Therefore ...' / 'Thus ...' lines (last match)
      4. Last non-empty line (fallback)
    """
    # 1. Boxed
    boxed = extract_boxed_answer(step_a_output)
    if boxed:
        return boxed

    # 2. Explicit answer line (scan bottom-up, take first hit)
    for line in reversed(step_a_output.strip().splitlines()):
        m = _ANSWER_LINE.search(line)
        if m:
            ans = m.group(1).strip().rstrip(".")
            # Strip surrounding $...$ if present
            if ans.startswith("$") and ans.endswith("$"):
                ans = ans[1:-1].strip()
            return ans

    # 3. Therefore/thus line
    for line in reversed(step_a_output.strip().splitlines()):
        m = _THEREFORE_LINE.search(line)
        if m:
            ans = m.group(1).strip().rstrip(".")
            if ans.startswith("$") and ans.endswith("$"):
                ans = ans[1:-1].strip()
            return ans

    # 4. Last non-empty line
    lines = [l.strip() for l in step_a_output.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def verify_step_a(step_a_output: str, gold_answer: str) -> dict:
    """Check if Step A produced the correct answer."""
    extracted = extract_step_a_answer(step_a_output)
    return {
        "correct": answers_match(extracted, gold_answer),
        "extracted_answer": extracted,
        "gold_answer_normalized": normalize_answer(gold_answer),
    }

#  ROUGE-L metrics

def _rouge_l(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    return _rouge.score(a, b)["rougeL"].fmeasure


def _split_steps(text: str) -> list[str]:
    """Split text into numbered steps, falling back to paragraphs or sentences."""
    parts = re.split(r"(?:^|\n)\s*(?:step\s*)?\d+\s*[.:)\-]\s*", text, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        return parts
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def solution_revelation_ratio(hints: str, full_solution: str) -> float:
    """ROUGE-L(hints, full solution). Higher = more leaked."""
    return _rouge_l(hints, full_solution)


def final_step_similarity(hints: str, full_solution: str) -> float:
    """ROUGE-L(hints, last solution step). Captures near-leaks."""
    steps = _split_steps(full_solution)
    return _rouge_l(hints, steps[-1]) if steps else 0.0


def intermediate_step_coverage(hints: str, full_solution: str) -> dict:
    """
    Per-step ROUGE-L breakdown.
    Returns reasoning_coverage (steps 1..K-1) and final_step_coverage (step K).
    """
    sol_steps = _split_steps(full_solution)
    hint_steps = _split_steps(hints)

    if not sol_steps or not hint_steps:
        return {"reasoning_coverage": 0.0, "final_step_coverage": 0.0, "per_step": []}

    per_step = [max(_rouge_l(h, s) for h in hint_steps) for s in sol_steps]

    K = len(sol_steps)
    reasoning_cov = sum(per_step[:-1]) / (K - 1) if K > 1 else 0.0

    return {
        "reasoning_coverage": reasoning_cov,
        "final_step_coverage": per_step[-1],
        "per_step": per_step,
    }
