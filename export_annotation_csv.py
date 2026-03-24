"""
export_annotation_csv.py — Export human-evaluation CSVs.

Samples N records per system from the JSONL results and writes one CSV
per system with columns: question_id, question, hint.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ALL_SYSTEMS, RESULTS_DIR

ANNOTATION_DIR = RESULTS_DIR / "annotation"


def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_records(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Deterministically sample n records (or return all if fewer exist)."""
    rng = random.Random(seed)
    if n >= len(records):
        return records
    return rng.sample(records, n)


def export_csv(records: list[dict], output_path: Path) -> None:
    """Write annotation CSV with question_id, attack_id, question, and hint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question_id", "attack_id", "question", "hint"]
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "question_id": rec.get("question_idx", ""),
                "attack_id": rec.get("attack_id", ""),
                "question": rec.get("problem", ""),
                "hint": rec.get("output", ""),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Export annotation CSVs for human evaluation."
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of samples per system (default: 100).",
    )
    parser.add_argument(
        "--systems", nargs="+", default=None, choices=ALL_SYSTEMS,
        help="Which systems to export (default: all available).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Which model results to export (default: all available).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Path to results directory.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling.",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    out_dir = results_dir / "annotation"

    # Discover model directories
    model_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name != "annotation")
    if args.models:
        model_dirs = [d for d in model_dirs if d.name in args.models]

    total_exported = 0

    for model_dir in model_dirs:
        model_id = model_dir.name
        system_files = sorted(model_dir.glob("*.jsonl"))

        for sf in system_files:
            system_id = sf.stem
            # Skip evaluated files
            if system_id.endswith("_evaluated"):
                continue
            if args.systems and system_id not in args.systems:
                continue

            records = load_records(sf)
            if not records:
                print(f"  {model_id}/{system_id}: no records, skipping.")
                continue

            sampled = sample_records(records, args.n, args.seed)
            csv_path = out_dir / model_id / f"{system_id}_annotation.csv"
            export_csv(sampled, csv_path)
            total_exported += 1
            print(f"  {model_id}/{system_id}: {len(sampled)} samples → {csv_path}")

    if total_exported == 0:
        print("No results found to export.")
    else:
        print(f"\nExported {total_exported} annotation CSVs to {out_dir}")


if __name__ == "__main__":
    main()
