"""Evaluate experiment outputs: annotate each record with leakage and compliance metrics, then summarize."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    ALL_SYSTEMS,
    RESULTS_DIR,
    iter_jsonl_objects,
    load_config,
    resolve_config_path,
)
from src.evaluation import (
    bootstrap_ci,
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


def load_results(path: Path) -> list[dict]:
    """Load JSONL result records, skipping damaged lines when needed."""
    return [
        record
        for _, record in iter_jsonl_objects(path, logger=logger, label=str(path))
    ]


def mean_metric(records: list[dict], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else None


def annotate_record(record: dict) -> dict:
    """Add evaluation metrics to one generated record."""
    output = record.get("output", "")
    gold = record.get("gold_answer", "")
    problem = record.get("problem", "")
    step_a = record.get("step_a_output")

    leak = detect_leakage(output, gold, problem, step_a_output=step_a)
    record["eval_leaked"] = leak["leaked"]
    record["eval_leak_tier"] = leak["leak_tier"]
    record["eval_gold_substring_match"] = leak["gold_substring_match"]
    record["eval_marker_matches"] = leak["marker_matches"]
    record["eval_explicit_answer_match"] = leak["explicit_answer_match"]
    record["eval_near_leak_fss"] = leak["near_leak_fss"]
    record["eval_prompt_contains_gold"] = leak["prompt_contains_gold"]

    compliance = check_step_compliance(output, gold, problem, step_a_output=step_a)
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

    # Use every generated record. Earlier versions excluded records when the
    # problem text contained the gold answer span, but that incorrectly removed
    # ordinary math problems such as "2x + 3 = 4" when the answer was "3".
    clean_records = records
    n_poisoned = 0

    step_a_values = [
        record["eval_step_a_correct"]
        for record in clean_records
        if record.get("eval_step_a_correct") is not None
    ]

    if step_a_values:
        step_a_accuracy = sum(step_a_values) / len(step_a_values)
        filtered_records = [
            record for record in clean_records if record.get("eval_step_a_correct") is True
        ]
    else:
        step_a_accuracy = None
        filtered_records = clean_records

    tiers = [r.get("eval_leak_tier", "clean") for r in clean_records]
    n = len(clean_records)
    explicit_flags = [t == "explicit_leak" for t in tiers]
    format_flags = [t == "format_violation" for t in tiers]
    answer_giving_flags = [t in ("explicit_leak", "format_violation") for t in tiers]

    explicit_ci = bootstrap_ci(explicit_flags)

    summary = {
        "n": len(records),
        "n_prompt_contains_gold": n_poisoned,
        "n_clean": n,
        "n_after_step_a_filter": len(filtered_records),
        "explicit_leak_rate": explicit_ci,
        "format_violation_rate": bootstrap_ci(format_flags),
        "answer_giving_rate": bootstrap_ci(answer_giving_flags),
        "leakage_rate": explicit_ci,
        "compliance_rate": bootstrap_ci(
            [r.get("eval_compliant", False) for r in clean_records]
        ),
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
    for record in clean_records:
        by_category[record.get("attack_category", "unknown")].append(record)

    summary["by_attack_category"] = {
        category: {
            "n": len(cat_recs),
            "explicit_leak_rate": bootstrap_ci(
                [r.get("eval_leak_tier") == "explicit_leak" for r in cat_recs]
            ),
        }
        for category, cat_recs in sorted(by_category.items())
    }
    return summary


def format_metric(value: float | dict | None, percent: bool = False) -> str:
    if value is None:
        return "  -  "
    if isinstance(value, dict) and "mean" in value:
        m = value["mean"]
        lo, hi = value["ci_lo"], value["ci_hi"]
        if percent:
            return f"{m*100:5.1f}% [{lo*100:.1f},{hi*100:.1f}]"
        return f"{m:.3f} [{lo:.3f},{hi:.3f}]"
    return f"{value * 100:5.1f}%" if percent else f"{value:5.3f}"


def _rate_point(value: float | dict | None) -> float | None:
    """Extract point estimate from either a scalar or a bootstrap_ci dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("mean")
    return value


def print_summary_table(all_aggs: dict[str, dict]) -> None:
    """Print a comparison table across all evaluated conditions."""
    header = (
        f"{'Condition':<25}  {'N':>5}  {'Explicit%':>18}  "
        f"{'FmtViol%':>18}  {'AnsGive%':>18}  {'Comply%':>18}  "
        f"{'SRR':>5}  {'FSS':>5}  {'ReaCov':>6}  {'FinCov':>6}  {'SA-Acc':>6}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for condition, agg in sorted(all_aggs.items()):
        print(
            f"{condition:<25}  {agg.get('n_clean', agg['n']):>5}  "
            f"{format_metric(agg.get('explicit_leak_rate'), True):>18}  "
            f"{format_metric(agg.get('format_violation_rate'), True):>18}  "
            f"{format_metric(agg.get('answer_giving_rate'), True):>18}  "
            f"{format_metric(agg.get('compliance_rate'), True):>18}  "
            f"{format_metric(agg.get('solution_revelation_ratio')):>5}  "
            f"{format_metric(agg.get('final_step_similarity')):>5}  "
            f"{format_metric(agg.get('reasoning_coverage')):>6}  "
            f"{format_metric(agg.get('final_step_coverage')):>6}  "
            f"{format_metric(agg.get('step_a_accuracy'), True):>6}"
        )

    print("=" * len(header))
    print("\nExplicit Leak Rate by Attack Category")
    for condition, agg in sorted(all_aggs.items()):
        categories = agg.get("by_attack_category", {})
        if not categories:
            continue
        parts = []
        for category, values in categories.items():
            rate = _rate_point(values.get("explicit_leak_rate"))
            if rate is not None:
                parts.append(f"{category}: {rate * 100:.0f}%")
        print(f"  {condition:<20}  {', '.join(parts)}")


def save_summary_csv(all_aggs: dict[str, dict], path: Path) -> None:
    """Save summary metrics as CSV."""
    fields = [
        "condition",
        "n",
        "n_prompt_contains_gold",
        "n_clean",
        "explicit_leak_rate",
        "explicit_leak_ci_lo",
        "explicit_leak_ci_hi",
        "format_violation_rate",
        "answer_giving_rate",
        "answer_giving_ci_lo",
        "answer_giving_ci_hi",
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
            elr = agg.get("explicit_leak_rate", {})
            agr = agg.get("answer_giving_rate", {})
            row = {
                "condition": condition,
                "n": agg.get("n"),
                "n_prompt_contains_gold": agg.get("n_prompt_contains_gold", 0),
                "n_clean": agg.get("n_clean"),
                "explicit_leak_rate": _rate_point(elr),
                "explicit_leak_ci_lo": elr.get("ci_lo") if isinstance(elr, dict) else None,
                "explicit_leak_ci_hi": elr.get("ci_hi") if isinstance(elr, dict) else None,
                "format_violation_rate": _rate_point(agg.get("format_violation_rate")),
                "answer_giving_rate": _rate_point(agr),
                "answer_giving_ci_lo": agr.get("ci_lo") if isinstance(agr, dict) else None,
                "answer_giving_ci_hi": agr.get("ci_hi") if isinstance(agr, dict) else None,
                "leakage_rate": _rate_point(agg.get("leakage_rate")),
                "compliance_rate": _rate_point(agg.get("compliance_rate")),
                "solution_revelation_ratio": agg.get("solution_revelation_ratio"),
                "final_step_similarity": agg.get("final_step_similarity"),
                "reasoning_coverage": agg.get("reasoning_coverage"),
                "final_step_coverage": agg.get("final_step_coverage"),
                "step_a_accuracy": agg.get("step_a_accuracy"),
            }
            writer.writerow(row)
    logger.info("Summary CSV saved to %s", path)


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
        config_data, config_base = load_config(args.config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    results_dir: Path = (
        resolve_config_path(config_data["results_dir"], config_base)
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
