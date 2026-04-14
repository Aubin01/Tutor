# paired_significance.py -- Post-evaluation paired statistical tests.
#
# This consumes *_evaluated.jsonl files produced by scripts/evaluate_results.py.
# It does not recompute leakage, compliance, or ROUGE metrics.
#
# Tests:
#   - record aggregation: exact McNemar test over aligned attack records
#   - question aggregation: Wilcoxon signed-rank test over per-question counts
#
# All reported p-values are Holm-Bonferroni corrected across generated rows.
#
# Usage:
#   python scripts/paired_significance.py --results-dir results

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ALL_MODELS, ALL_SYSTEMS, RESULTS_DIR, iter_jsonl_objects, load_config, resolve_config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _case_key(record: dict[str, Any]) -> tuple[str, str, str]:
    question_key = str(record.get("question_uid") or f"idx:{record.get('question_idx')}")
    attack_id = str(record.get("attack_id"))
    attack_category = str(record.get("attack_category", ""))
    return question_key, attack_id, attack_category


def _evaluated_path(results_dir: Path, model_id: str, system_id: str) -> Path:
    return results_dir / model_id / f"{system_id}_evaluated.jsonl"


def _load_evaluated_records(path: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], int]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_keys = 0
    for _, record in iter_jsonl_objects(path, logger=logger, label=str(path)):
        key = _case_key(record)
        if key in records:
            duplicate_keys += 1
        records[key] = record
    return records, duplicate_keys


def _extract_metric_success(record: dict[str, Any], metric: str) -> bool | None:
    if metric == "leakage":
        leaked = record.get("eval_leaked")
        if leaked is None:
            return None
        return not bool(leaked)
    if metric == "compliance":
        compliant = record.get("eval_compliant")
        if compliant is None:
            return None
        return bool(compliant)
    raise ValueError(f"Unsupported metric: {metric}")


def _extract_metric_rate(record: dict[str, Any], metric: str) -> float | None:
    if metric == "leakage":
        leaked = record.get("eval_leaked")
        return None if leaked is None else float(bool(leaked))
    if metric == "compliance":
        compliant = record.get("eval_compliant")
        return None if compliant is None else float(bool(compliant))
    raise ValueError(f"Unsupported metric: {metric}")


def _log_binom_pmf_half(n: int, k: int) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - n * math.log(2.0)
    )


def _binom_cdf_half(n: int, k: int) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0

    log_terms = [_log_binom_pmf_half(n, i) for i in range(k + 1)]
    max_log = max(log_terms)
    scaled = math.fsum(math.exp(term - max_log) for term in log_terms)
    return math.exp(max_log) * scaled


def exact_mcnemar_pvalue(candidate_better: int, reference_better: int) -> float:
    """Two-sided exact McNemar test using the binomial distribution.

    Counts how often only one system got it right (discordant pairs),
    then tests if the split is different from 50/50.
    """
    discordant_total = candidate_better + reference_better
    if discordant_total == 0:
        return 1.0

    smaller_tail = _binom_cdf_half(discordant_total, min(candidate_better, reference_better))
    return min(1.0, 2.0 * smaller_tail)


def _normal_two_sided_pvalue(z_score: float) -> float:
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def wilcoxon_signed_rank_pvalue(differences: list[float]) -> tuple[float, dict[str, float]]:
    """Wilcoxon signed-rank test with tie correction.

    Ranks the absolute differences, sums positive and negative ranks,
    and uses a normal approximation for the p-value.
    """
    nonzero = [diff for diff in differences if diff != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0, {"n_nonzero": 0.0, "w_plus": 0.0, "w_minus": 0.0, "z_score": 0.0}

    indexed = sorted((abs(diff), idx) for idx, diff in enumerate(nonzero))
    ranks = [0.0] * n
    tie_sizes: list[int] = []
    i = 0
    next_rank = 1

    while i < n:
        j = i + 1
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1

        tie_size = j - i
        avg_rank = (next_rank + next_rank + tie_size - 1) / 2.0
        for _, original_idx in indexed[i:j]:
            ranks[original_idx] = avg_rank

        tie_sizes.append(tie_size)
        next_rank += tie_size
        i = j

    w_plus = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    total_rank_sum = n * (n + 1) / 2.0
    w_minus = total_rank_sum - w_plus

    mean_w = n * (n + 1) / 4.0
    rank_square_sum = n * (n + 1) * (2 * n + 1) / 6.0
    tie_correction = sum(size**3 - size for size in tie_sizes if size > 1) / 12.0
    variance_w = (rank_square_sum - tie_correction) / 4.0

    if variance_w <= 0:
        return 1.0, {
            "n_nonzero": float(n),
            "w_plus": w_plus,
            "w_minus": w_minus,
            "z_score": 0.0,
        }

    continuity = 0.5 if w_plus != mean_w else 0.0
    z_score = (w_plus - mean_w - math.copysign(continuity, w_plus - mean_w)) / math.sqrt(variance_w)
    pvalue = _normal_two_sided_pvalue(z_score)
    return pvalue, {
        "n_nonzero": float(n),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "z_score": z_score,
    }


def holm_adjust(rows: list[dict[str, Any]], pvalue_key: str, output_key: str) -> None:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    indexed = sorted(
        ((row[pvalue_key], idx) for idx, row in enumerate(rows)),
        key=lambda item: item[0],
    )
    adjusted = [0.0] * len(rows)
    running_max = 0.0
    m = len(rows)

    for rank, (pvalue, original_idx) in enumerate(indexed, 1):
        scaled = min(1.0, (m - rank + 1) * pvalue)
        running_max = max(running_max, scaled)
        adjusted[original_idx] = running_max

    for row, value in zip(rows, adjusted):
        row[output_key] = value


def paired_test(
    reference_records: dict[tuple[str, str, str], dict[str, Any]],
    candidate_records: dict[tuple[str, str, str], dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    """Run McNemar's test on aligned record pairs for one metric."""
    shared_keys = sorted(set(reference_records) & set(candidate_records))

    reference_values: list[bool] = []
    candidate_values: list[bool] = []
    reference_rates: list[float] = []
    candidate_rates: list[float] = []

    for key in shared_keys:
        ref_record = reference_records[key]
        cand_record = candidate_records[key]

        ref_success = _extract_metric_success(ref_record, metric)
        cand_success = _extract_metric_success(cand_record, metric)
        ref_rate = _extract_metric_rate(ref_record, metric)
        cand_rate = _extract_metric_rate(cand_record, metric)

        if None in (ref_success, cand_success, ref_rate, cand_rate):
            continue

        reference_values.append(bool(ref_success))
        candidate_values.append(bool(cand_success))
        reference_rates.append(float(ref_rate))
        candidate_rates.append(float(cand_rate))

    n_pairs = len(reference_values)
    if n_pairs == 0:
        raise ValueError(f"No aligned records available for metric {metric}")

    candidate_better = sum(
        1
        for ref_success, cand_success in zip(reference_values, candidate_values)
        if not ref_success and cand_success
    )
    reference_better = sum(
        1
        for ref_success, cand_success in zip(reference_values, candidate_values)
        if ref_success and not cand_success
    )

    discordant_total = candidate_better + reference_better
    pvalue = exact_mcnemar_pvalue(candidate_better, reference_better)

    reference_rate = sum(reference_rates) / n_pairs
    candidate_rate = sum(candidate_rates) / n_pairs
    raw_success_delta = (sum(candidate_values) - sum(reference_values)) / n_pairs

    if metric == "leakage":
        delta_rate = candidate_rate - reference_rate
        improvement_rate = -delta_rate
    else:
        delta_rate = candidate_rate - reference_rate
        improvement_rate = delta_rate

    return {
        "metric": metric,
        "aggregation": "record",
        "test": "mcnemar_exact",
        "n_pairs": n_pairs,
        "reference_rate": reference_rate,
        "candidate_rate": candidate_rate,
        "delta_rate": delta_rate,
        "improvement_rate": improvement_rate,
        "candidate_better": candidate_better,
        "reference_better": reference_better,
        "discordant_total": discordant_total,
        "success_delta": raw_success_delta,
        "p_value": pvalue,
    }


def aggregate_question_counts(
    reference_records: dict[tuple[str, str, str], dict[str, Any]],
    candidate_records: dict[tuple[str, str, str], dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    """Aggregate per-question counts and run Wilcoxon signed-rank test."""
    shared_keys = set(reference_records) & set(candidate_records)
    per_question: dict[str, dict[str, float]] = {}

    for key in shared_keys:
        question_key = key[0]
        ref_rate = _extract_metric_rate(reference_records[key], metric)
        cand_rate = _extract_metric_rate(candidate_records[key], metric)
        if ref_rate is None or cand_rate is None:
            continue

        bucket = per_question.setdefault(
            question_key,
            {
                "reference_sum": 0.0,
                "candidate_sum": 0.0,
                "attack_count": 0.0,
            },
        )
        bucket["reference_sum"] += ref_rate
        bucket["candidate_sum"] += cand_rate
        bucket["attack_count"] += 1.0

    question_keys = sorted(per_question)
    if not question_keys:
        raise ValueError(f"No aligned question aggregates available for metric {metric}")

    reference_counts: list[float] = []
    candidate_counts: list[float] = []
    improvement_differences: list[float] = []
    attack_counts: list[float] = []

    for question_key in question_keys:
        bucket = per_question[question_key]
        reference_sum = bucket["reference_sum"]
        candidate_sum = bucket["candidate_sum"]
        attacks = bucket["attack_count"]

        reference_counts.append(reference_sum)
        candidate_counts.append(candidate_sum)
        attack_counts.append(attacks)

        if metric == "leakage":
            improvement_differences.append(reference_sum - candidate_sum)
        else:
            improvement_differences.append(candidate_sum - reference_sum)

    pvalue, wilcoxon_stats = wilcoxon_signed_rank_pvalue(improvement_differences)
    n_questions = len(question_keys)
    attacks_per_question = sum(attack_counts) / n_questions
    reference_rate = sum(reference_counts) / sum(attack_counts)
    candidate_rate = sum(candidate_counts) / sum(attack_counts)
    mean_count_delta = sum(improvement_differences) / n_questions
    mean_rate_delta = mean_count_delta / attacks_per_question if attacks_per_question else 0.0

    return {
        "metric": metric,
        "aggregation": "question",
        "test": "wilcoxon_signed_rank",
        "n_pairs": n_questions,
        "reference_rate": reference_rate,
        "candidate_rate": candidate_rate,
        "delta_rate": -mean_rate_delta if metric == "leakage" else mean_rate_delta,
        "improvement_rate": mean_rate_delta,
        "reference_mean_count": sum(reference_counts) / n_questions,
        "candidate_mean_count": sum(candidate_counts) / n_questions,
        "mean_count_delta": mean_count_delta,
        "attacks_per_question": attacks_per_question,
        "candidate_better": sum(1 for diff in improvement_differences if diff > 0),
        "reference_better": sum(1 for diff in improvement_differences if diff < 0),
        "discordant_total": sum(1 for diff in improvement_differences if diff != 0),
        "success_delta": mean_rate_delta,
        "p_value": pvalue,
        "wilcoxon_n_nonzero": int(wilcoxon_stats["n_nonzero"]),
        "wilcoxon_w_plus": wilcoxon_stats["w_plus"],
        "wilcoxon_w_minus": wilcoxon_stats["w_minus"],
        "wilcoxon_z": wilcoxon_stats["z_score"],
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "reference_system",
        "candidate_system",
        "aggregation",
        "test",
        "metric",
        "n_pairs",
        "reference_rate",
        "candidate_rate",
        "delta_rate",
        "improvement_rate",
        "reference_mean_count",
        "candidate_mean_count",
        "mean_count_delta",
        "attacks_per_question",
        "candidate_better",
        "reference_better",
        "discordant_total",
        "p_value",
        "p_value_holm",
        "significant_0_05",
        "wilcoxon_n_nonzero",
        "wilcoxon_w_plus",
        "wilcoxon_w_minus",
        "wilcoxon_z",
        "duplicates_reference",
        "duplicates_candidate",
        "missing_reference",
        "missing_candidate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'Comparison':<26}  {'Agg':<8}  {'Metric':<10}  {'N':>5}  {'Ref%':>7}  {'Cand%':>7}  "
        f"{'Imp pp':>7}  {'Cand+':>6}  {'Ref+':>6}  {'p':>10}  {'p_holm':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        comparison = f"{row['model']}:{row['reference_system']}->{row['candidate_system']}"
        print(
            f"{comparison:<26}  "
            f"{row['aggregation']:<8}  "
            f"{row['metric']:<10}  "
            f"{row['n_pairs']:>5}  "
            f"{row['reference_rate'] * 100:>6.2f}%  "
            f"{row['candidate_rate'] * 100:>6.2f}%  "
            f"{row['improvement_rate'] * 100:>+6.2f}  "
            f"{row['candidate_better']:>6}  "
            f"{row['reference_better']:>6}  "
            f"{row['p_value']:>10.3g}  "
            f"{row['p_value_holm']:>10.3g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired significance tests on evaluated outputs.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS)
    parser.add_argument(
        "--reference-system",
        default="B1",
        choices=ALL_SYSTEMS,
        help="Reference system to compare against.",
    )
    parser.add_argument(
        "--compare-systems",
        nargs="+",
        default=None,
        choices=ALL_SYSTEMS,
        help="Candidate systems to compare to the reference. Defaults to all except the reference.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["leakage", "compliance"],
        choices=["leakage", "compliance"],
    )
    parser.add_argument(
        "--aggregation",
        default="record",
        choices=["record", "question"],
        help="Compare record-level binary outcomes or question-level aggregated counts.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
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
    if args.models is not None:
        models = args.models
    if models is None:
        models = list(ALL_MODELS)

    compare_systems = args.compare_systems
    if compare_systems is None:
        compare_systems = [system for system in ALL_SYSTEMS if system != args.reference_system]

    rows: list[dict[str, Any]] = []

    for model_id in models:
        reference_path = _evaluated_path(results_dir, model_id, args.reference_system)
        if not reference_path.exists():
            logger.warning("Missing reference file: %s", reference_path)
            continue

        reference_records, reference_duplicates = _load_evaluated_records(reference_path)

        for candidate_system in compare_systems:
            if candidate_system == args.reference_system:
                continue

            candidate_path = _evaluated_path(results_dir, model_id, candidate_system)
            if not candidate_path.exists():
                logger.warning("Missing candidate file: %s", candidate_path)
                continue

            candidate_records, candidate_duplicates = _load_evaluated_records(candidate_path)
            reference_only = len(set(reference_records) - set(candidate_records))
            candidate_only = len(set(candidate_records) - set(reference_records))

            for metric in args.metrics:
                if args.aggregation == "record":
                    result = paired_test(reference_records, candidate_records, metric=metric)
                else:
                    result = aggregate_question_counts(
                        reference_records,
                        candidate_records,
                        metric=metric,
                    )
                result.update(
                    {
                        "model": model_id,
                        "reference_system": args.reference_system,
                        "candidate_system": candidate_system,
                        "duplicates_reference": reference_duplicates,
                        "duplicates_candidate": candidate_duplicates,
                        "missing_reference": reference_only,
                        "missing_candidate": candidate_only,
                    }
                )
                rows.append(result)

    if not rows:
        logger.warning("No paired comparisons were generated.")
        return

    holm_adjust(rows, "p_value", "p_value_holm")
    for row in rows:
        row["significant_0_05"] = row["p_value_holm"] < 0.05

    rows.sort(key=lambda row: (row["model"], row["metric"], row["candidate_system"]))

    _print_table(rows)

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = results_dir / f"paired_significance_{args.reference_system}_{args.aggregation}.csv"
    output_json = args.output_json
    if output_json is None:
        output_json = results_dir / f"paired_significance_{args.reference_system}_{args.aggregation}.json"

    _write_csv(rows, output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    logger.info("Paired significance CSV saved to %s", output_csv)
    logger.info("Paired significance JSON saved to %s", output_json)


if __name__ == "__main__":
    main()
