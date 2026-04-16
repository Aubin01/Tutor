#!/usr/bin/env python3
"""Download and format math problems from the MATH benchmark into a local JSON file.

Default source: Hugging Face dataset 'qwedsacf/competition_math'.
Output schema: problem, solution, level, type.
Produces a 500-problem stratified sample by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_HF_DATASET_ID = "qwedsacf/competition_math"
DEFAULT_HF_SPLIT = "train"
REQUIRED_FIELDS = {"problem", "solution", "level"}
MATH_SIGNAL_RE = re.compile(r"(\\frac|\\sqrt|\\pi|\\cdot|\\times|=|\^|\d)")
LEVEL_RE = re.compile(r"(\d+)")

def normalize_record(
    record: dict[str, Any],
    *,
    split: str,
    source_name: str,
) -> dict[str, Any]:
    problem_type = record.get("type") or ""
    normalized = {
        "problem": record["problem"],
        "solution": record["solution"],
        "level": record["level"],
        "type": problem_type,
        "split": split,
        "source_file": source_name,
    }
    return normalized


def load_records_from_hf(
    dataset_id: str,
    dataset_split: str,
    cache_dir: Path | None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `datasets`. Run ./bin/install first."
        ) from exc

    dataset = load_dataset(
        dataset_id,
        split=dataset_split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No records found in Hugging Face dataset {dataset_id} split {dataset_split}."
        )

    records: list[dict[str, Any]] = []
    split_label = f"hf:{dataset_split}"
    for idx, row in enumerate(dataset):
        row_data = dict(row)
        missing = sorted(REQUIRED_FIELDS - set(row_data))
        if missing:
            joined = ", ".join(missing)
            raise KeyError(
                f"{dataset_id} row {idx} is missing required keys: {joined}"
            )
        records.append(
            normalize_record(
                row_data,
                split=split_label,
                source_name=f"{dataset_id}:{dataset_split}:{idx}",
            )
        )

    return records


def stable_record_id(record: dict[str, Any]) -> str:
    """Stable id from immutable content for deterministic ordering."""
    payload = f"{record.get('problem', '')}\n||\n{record.get('solution', '')}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def record_sort_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Canonical sort key for deterministic selection and output."""
    return (
        str(record.get("type", "")),
        str(record.get("level", "")),
        str(record.get("split", "")),
        str(record.get("source_file", "")),
        stable_record_id(record),
    )


def parse_level_number(level_value: Any) -> int | None:
    text = str(level_value).strip()
    match = LEVEL_RE.search(text)
    if not match:
        return None
    value = int(match.group(1))
    if 1 <= value <= 5:
        return value
    return None


def quality_score(record: dict[str, Any]) -> int:
    """
    Objective quality score used for within-stratum ranking.

    Score components (0..100):
      +30 solution contains a boxed final answer marker
      +20 solution length is in a usable tutoring range
      +15 problem length is in a usable tutoring range
      +15 level is parseable and in [1, 5]
      +10 type label is present
      +10 math signal appears in problem and solution text
    """
    problem = str(record.get("problem", "")).strip()
    solution = str(record.get("solution", "")).strip()
    level = record.get("level", "")
    problem_type = str(record.get("type", "")).strip()

    score = 0

    has_boxed = "\\boxed{" in solution
    if has_boxed:
        score += 30

    problem_len = len(problem)
    solution_len = len(solution)
    if 40 <= problem_len <= 5000:
        score += 15
    if 80 <= solution_len <= 12000:
        score += 20

    if parse_level_number(level) is not None:
        score += 15

    if problem_type:
        score += 10

    if MATH_SIGNAL_RE.search(problem) and MATH_SIGNAL_RE.search(solution):
        score += 10

    return score


def quality_sort_key(record: dict[str, Any]) -> tuple[int, tuple[str, str, str, str, str]]:
    """Higher quality first, then stable deterministic tie-break."""
    return (-int(record.get("quality_score", 0)), record_sort_key(record))


def stratified_deterministic_sample(
    records: list[dict[str, Any]],
    sample_size: int,
) -> list[dict[str, Any]]:
    """
    Deterministic proportional stratification by (type, level).

    Allocation:
      1. floor(sample_size * stratum_count / total_count)
      2. assign remaining slots to largest fractional remainders
         (ties resolved deterministically).
    """
    if sample_size <= 0 or sample_size >= len(records):
        return sorted(records, key=quality_sort_key)

    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("type", "")), str(record.get("level", "")))
        strata.setdefault(key, []).append(record)

    for key in strata:
        strata[key] = sorted(strata[key], key=quality_sort_key)

    total = len(records)
    quotas: dict[tuple[str, str], int] = {}
    remainder_rank: list[tuple[float, int, tuple[str, str]]] = []
    allocated = 0

    for key in sorted(strata):
        group_size = len(strata[key])
        raw_quota = sample_size * group_size / total
        base_quota = int(raw_quota)
        quotas[key] = base_quota
        allocated += base_quota
        remainder_rank.append((raw_quota - base_quota, group_size, key))

    shortfall = sample_size - allocated
    remainder_rank.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for _, _, key in remainder_rank[:shortfall]:
        quotas[key] += 1

    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        take = quotas[key]
        if take > 0:
            selected.extend(strata[key][:take])

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic stratified pilot dataset from an official MATH "
            "benchmark download."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/math.json"),
        help="Where to write the pilot JSON output (default: Data/math.json).",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_HF_DATASET_ID,
        help=(
            "Hugging Face dataset id to download "
            f"(default: {DEFAULT_HF_DATASET_ID})."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        default=DEFAULT_HF_SPLIT,
        help=(
            "Hugging Face split to use "
            f"(default: {DEFAULT_HF_SPLIT})."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory for downloaded dataset files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help=(
            "Target pilot size. <=0 keeps all records. "
            "Default: 500 (stratified by type and level)."
        ),
    )
    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    records = load_records_from_hf(args.dataset_id, args.dataset_split, cache_dir)
    source_desc = f"Hugging Face {args.dataset_id} ({args.dataset_split})"

    for record in records:
        record["quality_score"] = quality_score(record)

    selected = stratified_deterministic_sample(records, args.sample_size)
    split_counts: Counter[str] = Counter(str(record.get("split", "")) for record in selected)
    type_counts: Counter[str] = Counter(str(record.get("type", "")) for record in selected)
    level_counts: Counter[str] = Counter(str(record.get("level", "")) for record in selected)
    quality_counts: Counter[int] = Counter(int(record.get("quality_score", 0)) for record in selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(selected)} records to {output_path} "
        f"(from {len(records)} source records)."
    )
    print(f"Data source: {source_desc}")
    print("Sampling policy: deterministic stratified by (type, level)")
    print("Split counts:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")
    print("Top levels:")
    for level, count in level_counts.most_common(10):
        print(f"  {level}: {count}")
    print("Top problem types:")
    for problem_type, count in type_counts.most_common(10):
        print(f"  {problem_type}: {count}")
    print("Quality score distribution (selected set):")
    for score, count in sorted(quality_counts.items(), reverse=True):
        print(f"  {score}: {count}")


if __name__ == "__main__":
    main()
