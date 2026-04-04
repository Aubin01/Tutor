"""
Evaluate raw JSONL experiment outputs and write annotated summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_SYSTEMS, RESULTS_DIR
from src.evaluation import (
    check_step_compliance,
    detect_leakage,
    final_step_similarity,
    intermediate_step_coverage,
    solution_revelation_ratio,
    verify_step_a,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_results(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def mean_metric(records: list[dict], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else None


def annotate_record(record: dict) -> dict:
    """Add evaluation metrics to one generated record."""
    output = record.get("output", "")
    gold = record.get("gold_answer", "")
    problem = record.get("problem", "")
    step_a = record.get("step_a_output")

    leak = detect_leakage(output, gold, problem)
    record["eval_leaked"] = leak["leaked"]
    record["eval_gold_substring_match"] = leak["gold_substring_match"]
    record["eval_marker_matches"] = leak["marker_matches"]
    record["eval_explicit_answer_match"] = leak["explicit_answer_match"]
    record["eval_prompt_contains_gold"] = leak["prompt_contains_gold"]

    compliance = check_step_compliance(output, gold, problem)
    record["eval_compliant"] = compliance["compliant"]
    record["eval_num_steps"] = compliance["num_steps"]
    record["eval_compliance_reasons"] = compliance["reasons"]

    if not step_a:
        record["eval_step_a_correct"] = None
        record["eval_solution_revelation_ratio"] = None
        record["eval_final_step_similarity"] = None
        record["eval_reasoning_coverage"] = None
        record["eval_final_step_coverage"] = None
        return record

    step_a_check = verify_step_a(step_a, gold)
    record["eval_step_a_correct"] = step_a_check["correct"]
    record["eval_step_a_extracted"] = step_a_check["extracted_answer"]
    record["eval_solution_revelation_ratio"] = solution_revelation_ratio(output, step_a)
    record["eval_final_step_similarity"] = final_step_similarity(output, step_a)

    coverage = intermediate_step_coverage(output, step_a)
    record["eval_reasoning_coverage"] = coverage["reasoning_coverage"]
    record["eval_final_step_coverage"] = coverage["final_step_coverage"]
    return record


def aggregate(records: list[dict]) -> dict:
    """Compute summary metrics for a set of annotated records."""
    if not records:
        return {}

    step_a_values = [
        record["eval_step_a_correct"]
        for record in records
        if record.get("eval_step_a_correct") is not None
    ]

    if step_a_values:
        step_a_accuracy = sum(step_a_values) / len(step_a_values)
        filtered_records = [
            record for record in records if record.get("eval_step_a_correct") is True
        ]
    else:
        step_a_accuracy = None
        filtered_records = records

    summary = {
        "n": len(records),
        "n_after_step_a_filter": len(filtered_records),
        "leakage_rate": mean_metric(records, "eval_leaked"),
        "compliance_rate": mean_metric(records, "eval_compliant"),
        "solution_revelation_ratio": mean_metric(
            filtered_records, "eval_solution_revelation_ratio"
        ),
        "final_step_similarity": mean_metric(
            filtered_records, "eval_final_step_similarity"
        ),
        "reasoning_coverage": mean_metric(filtered_records, "eval_reasoning_coverage"),
        "final_step_coverage": mean_metric(filtered_records, "eval_final_step_coverage"),
    }

    if step_a_accuracy is not None:
        summary["step_a_accuracy"] = step_a_accuracy
        summary["step_a_exclusion_rate"] = 1.0 - step_a_accuracy

    by_category: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_category[record.get("attack_category", "unknown")].append(record)

    summary["by_attack_category"] = {
        category: {
            "n": len(category_records),
            "leakage_rate": mean_metric(category_records, "eval_leaked"),
        }
        for category, category_records in sorted(by_category.items())
    }
    return summary


def format_metric(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "  -  "
    return f"{value * 100:5.1f}%" if percent else f"{value:5.3f}"


def print_summary_table(all_aggs: dict[str, dict]) -> None:
    """Print a comparison table across all evaluated conditions."""
    header = (
        f"{'Condition':<25}  {'N':>5}  {'Leak%':>6}  {'Comply%':>7}  "
        f"{'SRR':>5}  {'FSS':>5}  {'ReaCov':>6}  {'FinCov':>6}  {'SA-Acc':>6}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for condition, agg in sorted(all_aggs.items()):
        print(
            f"{condition:<25}  {agg['n']:>5}  "
            f"{format_metric(agg.get('leakage_rate'), True):>6}  "
            f"{format_metric(agg.get('compliance_rate'), True):>7}  "
            f"{format_metric(agg.get('solution_revelation_ratio')):>5}  "
            f"{format_metric(agg.get('final_step_similarity')):>5}  "
            f"{format_metric(agg.get('reasoning_coverage')):>6}  "
            f"{format_metric(agg.get('final_step_coverage')):>6}  "
            f"{format_metric(agg.get('step_a_accuracy'), True):>6}"
        )

    print("=" * len(header))
    print("\nLeakage Rate by Attack Category")
    for condition, agg in sorted(all_aggs.items()):
        categories = agg.get("by_attack_category", {})
        if not categories:
            continue
        parts = [
            f"{category}: {values['leakage_rate'] * 100:.0f}%"
            for category, values in categories.items()
            if values["leakage_rate"] is not None
        ]
        print(f"  {condition:<20}  {', '.join(parts)}")


def save_summary_csv(all_aggs: dict[str, dict], path: Path) -> None:
    """Save summary metrics as CSV."""
    fields = [
        "condition",
        "n",
        "leakage_rate",
        "compliance_rate",
        "solution_revelation_ratio",
        "final_step_similarity",
        "reasoning_coverage",
        "final_step_coverage",
        "step_a_accuracy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for condition, agg in sorted(all_aggs.items()):
            row = {
                "condition": condition,
                **{field: agg.get(field) for field in fields if field != "condition"},
            }
            writer.writerow(row)
    logger.info("Summary CSV saved to %s", path)


def _load_config(config_path: Path | None) -> tuple[dict[str, object], Path]:
    if config_path is None:
        return {}, Path(__file__).resolve().parent

    path = config_path.expanduser().resolve()
    if not path.exists():
        raise ValueError(f"config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")

    return data, path.parent


def _resolve_config_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Bare relative paths in config are resolved from project root
    # (e.g., "results" -> <repo>/results).
    # Use "./" or "../" in config when you explicitly want config-file-relative paths.
    if path.parts and path.parts[0] in (".", ".."):
        return (base_dir / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment results.")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON config file. CLI flags override config values.",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--systems", nargs="+", default=None, choices=ALL_SYSTEMS)
    args = parser.parse_args()

    try:
        config_data, config_base = _load_config(args.config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    results_dir: Path = (
        _resolve_config_path(config_data["results_dir"], config_base)
        if "results_dir" in config_data
        else args.results_dir
    )
    models = config_data.get("models")
    systems = config_data.get("systems")
    if args.models is not None:
        models = args.models
    if args.systems is not None:
        systems = args.systems

    all_aggs: dict[str, dict] = {}
    files_to_evaluate: list[tuple[str, str, Path]] = []

    if not results_dir.exists():
        logger.warning("Results directory does not exist: %s", results_dir)
        logger.warning("No results to evaluate.")
        return

    model_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    if models:
        model_dirs = [path for path in model_dirs if path.name in models]

    for model_dir in model_dirs:
        model_id = model_dir.name
        system_files = [
            path
            for path in sorted(model_dir.glob("*.jsonl"))
            if "_evaluated" not in path.stem
        ]

        for system_file in system_files:
            system_id = system_file.stem
            if systems and system_id not in systems:
                continue

            files_to_evaluate.append((model_id, system_id, system_file))

    if files_to_evaluate:
        logger.info("Evaluating %d file(s).", len(files_to_evaluate))

    for model_id, system_id, system_file in tqdm(
        files_to_evaluate,
        desc="Evaluating files",
        unit="file",
    ):
        condition = f"{model_id}/{system_id}"
        logger.info("Evaluating %s (%s)", condition, system_file.name)

        records = load_results(system_file)
        if not records:
            logger.warning("No records found for %s, skipping.", condition)
            continue

        annotated = [
            annotate_record(record)
            for record in tqdm(records, desc=condition, unit="record", leave=False)
        ]

        annotated_path = system_file.with_name(f"{system_id}_evaluated.jsonl")
        with open(annotated_path, "w", encoding="utf-8") as f:
            for record in annotated:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        all_aggs[condition] = aggregate(annotated)

    if not all_aggs:
        logger.warning("No results to evaluate.")
        return

    print_summary_table(all_aggs)

    csv_path = results_dir / "summary.csv"
    save_summary_csv(all_aggs, csv_path)

    json_path = results_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_aggs, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Summary JSON saved to %s", json_path)


if __name__ == "__main__":
    main()
