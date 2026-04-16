"""Test if tutor hints help a solver model.

Runs a solver (e.g. Phi-4-reasoning) on math problems with and without hints.
Compares solve rates to measure hint quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ALL_SYSTEMS, RESULTS_DIR, ModelConfig, load_config, resolve_config_path, iter_jsonl_objects, PROJECT_ROOT
from src.evaluation import verify_step_a
from src.pipeline import load_model

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROMPT_VERSION = "v2"
SOLVER_BACKEND = "huggingface"
DEFAULT_SOLVER_MODEL = "microsoft/Phi-4-reasoning"

SOLVER_SYSTEM = (
    "You are an expert math solver. Solve the math problem exactly. "
    "Put the final answer inside \\boxed{} on its own line."
)


def _solver_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return slug or "solver"


def _question_key(record: dict[str, Any]) -> str:
    question_uid = record.get("question_uid")
    if question_uid:
        return str(question_uid)
    return f"idx:{record.get('question_idx')}"


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _case_key(record: dict[str, Any]) -> str:
    return json.dumps(
        {
            "question": _question_key(record),
            "attack_id": str(record.get("attack_id")),
            "attack_category": str(record.get("attack_category", "")),
            "problem_hash": _short_hash(str(record.get("problem", ""))),
            "query_hash": _short_hash(str(record.get("attack_prompt", ""))),
        },
        sort_keys=True,
    )


def _no_hint_key(record: dict[str, Any]) -> str:
    return json.dumps(
        {
            "question": _question_key(record),
            "problem_hash": _short_hash(str(record.get("problem", ""))),
        },
        sort_keys=True,
    )


def _sample_key_parts(record: dict[str, Any]) -> dict[str, str]:
    return {
        "question_key": _question_key(record),
        "attack_id": str(record.get("attack_id")),
        "attack_category": str(record.get("attack_category", "")),
    }


def _sample_key(record: dict[str, Any]) -> str:
    return json.dumps(_sample_key_parts(record), sort_keys=True)


def _normalize_sample_entry(entry: str | dict[str, Any]) -> str:
    if isinstance(entry, str):
        return entry

    question_key = (
        entry.get("question_key")
        or entry.get("question_uid")
        or (
            f"idx:{entry.get('question_idx')}"
            if entry.get("question_idx") is not None
            else None
        )
    )
    if question_key is None:
        raise ValueError(f"Sample entry is missing question_key/question_uid: {entry}")

    return json.dumps(
        {
            "question_key": str(question_key),
            "attack_id": str(entry.get("attack_id")),
            "attack_category": str(entry.get("attack_category", "")),
        },
        sort_keys=True,
    )


def _sample_file_entry(record: dict[str, Any]) -> dict[str, Any]:
    parts = _sample_key_parts(record)
    return {
        **parts,
        "question_uid": record.get("question_uid", ""),
        "question_idx": record.get("question_idx"),
        "attack_id": record.get("attack_id"),
        "attack_category": record.get("attack_category", ""),
        "level": record.get("level", ""),
        "type": record.get("type", ""),
        "problem_hash": _short_hash(str(record.get("problem", ""))),
        "query_hash": _short_hash(str(record.get("attack_prompt", ""))),
    }


def _source_key(model_id: str, system_id: str, ordinal: int) -> str:
    return f"{model_id}/{system_id}:{ordinal}"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records

    for _, record in iter_jsonl_objects(path, logger=logger, label=str(path)):
        value = record.get(key)
        if value is not None:
            records[str(value)] = record
    return records


def _load_result_records(path: Path) -> list[tuple[int, int, dict[str, Any]]]:
    return [
        (ordinal, line_no, record)
        for ordinal, (line_no, record) in enumerate(
            iter_jsonl_objects(path, logger=logger, label=str(path)),
            1,
        )
    ]


def _no_hint_prompt(record: dict[str, Any]) -> str:
    return (
        "Math Problem:\n"
        f"{record.get('problem', '')}\n\n"
        "Solve the math problem and provide the final answer in \\boxed{}."
    )


def _hinted_prompt(record: dict[str, Any]) -> str:
    return (
        "Math Problem:\n"
        f"{record.get('problem', '')}\n\n"
        "Tutor Hints:\n"
        f"{record.get('output', '')}\n\n"
        "Use the tutor information if it helps, but verify the reasoning yourself. "
        "Solve the math problem and provide the final answer in \\boxed{}."
    )


def _token_estimate(text: str) -> float:
    return len(text) / 4.0


def _input_tokens_no_hint(record: dict[str, Any]) -> float:
    return _token_estimate(SOLVER_SYSTEM) + _token_estimate(_no_hint_prompt(record))


def _input_tokens_hinted(record: dict[str, Any]) -> float:
    return _token_estimate(SOLVER_SYSTEM) + _token_estimate(_hinted_prompt(record))


def _check_solver_output(output: str, gold_answer: str) -> dict[str, Any]:
    check = verify_step_a(output, gold_answer)
    return {
        "correct": bool(check["correct"]),
        "extracted_answer": check["extracted_answer"],
        "gold_answer_normalized": check["gold_answer_normalized"],
    }


def _outcome(no_hint_correct: bool, hinted_correct: bool) -> str:
    if not no_hint_correct and hinted_correct:
        return "helped"
    if no_hint_correct and not hinted_correct:
        return "hurt"
    if no_hint_correct and hinted_correct:
        return "unchanged_correct"
    return "unchanged_wrong"


def _mean_bool(records: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute no-hint rate, hinted rate, and gain for a set of records."""
    no_hint_rate = _mean_bool(records, "eval_no_hint_solve_correct")
    hinted_rate = _mean_bool(records, "eval_hinted_solve_correct")
    hint_gain = (
        hinted_rate - no_hint_rate
        if no_hint_rate is not None and hinted_rate is not None
        else None
    )

    result = {
        "n": len(records),
        "no_hint_solve_rate": no_hint_rate,
        "hinted_solve_rate": hinted_rate,
        "hint_gain": hint_gain,
    }

    # McNemar's exact test on paired (no_hint_correct, hinted_correct)
    if HAS_STATSMODELS:
        paired = [
            (bool(r["eval_no_hint_solve_correct"]), bool(r["eval_hinted_solve_correct"]))
            for r in records
            if r.get("eval_no_hint_solve_correct") is not None
            and r.get("eval_hinted_solve_correct") is not None
        ]
        if paired:
            # 2×2 contingency: [[both_correct, helped], [hurt, both_wrong]]
            both_correct = sum(1 for a, b in paired if a and b)
            helped = sum(1 for a, b in paired if not a and b)      # no-hint wrong, hinted correct
            hurt = sum(1 for a, b in paired if a and not b)        # no-hint correct, hinted wrong
            both_wrong = sum(1 for a, b in paired if not a and not b)
            table = [[both_correct, hurt], [helped, both_wrong]]
            try:
                bunch = mcnemar(table, exact=True)
                result["mcnemar_p"] = bunch.pvalue
                result["mcnemar_statistic"] = bunch.statistic
            except Exception as exc:
                logger.warning("McNemar test failed: %s", exc)
            result["n_helped"] = helped
            result["n_hurt"] = hurt
            result["n_both_correct"] = both_correct
            result["n_both_wrong"] = both_wrong

    return result


def _format_percent(value: float | None) -> str:
    if value is None:
        return "  -  "
    return f"{value * 100:6.1f}%"


def _load_solver(args: argparse.Namespace):
    model_config = ModelConfig(
        model_id="general",
        backend=SOLVER_BACKEND,
        model_name=args.solver_model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return load_model(model_config)


def _format_pvalue(p: float | None) -> str:
    if p is None:
        return "  -  "
    if p < 0.001:
        return " <0.001"
    return f"{p:7.4f}"


def print_summary_table(all_aggs: dict[str, dict[str, Any]]) -> None:
    header = (
        f"{'Condition':<25}  {'N':>5}  {'NoHint%':>8}  "
        f"{'Hinted%':>8}  {'Gain':>8}  {'Helped':>7}  {'Hurt':>5}  {'McNem p':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for condition, agg in sorted(all_aggs.items()):
        print(
            f"{condition:<25}  {agg['n']:>5}  "
            f"{_format_percent(agg.get('no_hint_solve_rate')):>8}  "
            f"{_format_percent(agg.get('hinted_solve_rate')):>8}  "
            f"{_format_percent(agg.get('hint_gain')):>8}  "
            f"{agg.get('n_helped', '-'):>7}  "
            f"{agg.get('n_hurt', '-'):>5}  "
            f"{_format_pvalue(agg.get('mcnemar_p')):>8}"
        )
    print("=" * len(header))


def save_summary_csv(all_aggs: dict[str, dict[str, Any]], path: Path) -> None:
    fields = [
        "condition",
        "n",
        "no_hint_solve_rate",
        "hinted_solve_rate",
        "hint_gain",
        "n_helped",
        "n_hurt",
        "n_both_correct",
        "n_both_wrong",
        "mcnemar_statistic",
        "mcnemar_p",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for condition, agg in sorted(all_aggs.items()):
            writer.writerow(
                {
                    "condition": condition,
                    **{field: agg.get(field) for field in fields if field != "condition"},
                }
            )


def _filter_records_by_sample(
    source_records: list[tuple[int, int, dict[str, Any]]],
    sample_keys: set[str] | None,
) -> list[tuple[int, int, dict[str, Any]]]:
    if sample_keys is None:
        return source_records
    return [
        row
        for row in source_records
        if _sample_key(row[2]) in sample_keys
    ]


def _load_sample_keys(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("keys") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError(f"Sample file must contain a list or a dict with keys: {path}")

    sample_keys = {_normalize_sample_entry(entry) for entry in entries}
    if not sample_keys:
        raise ValueError(f"Sample file contains no usable keys: {path}")
    return sample_keys


def _build_balanced_sample(
    result_files: list[tuple[str, str, Path]],
    *,
    sample_size: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive when creating a sample file")

    records_by_file: list[dict[str, dict[str, Any]]] = []
    for _, _, source_path in result_files:
        by_key: dict[str, dict[str, Any]] = {}
        for _, _, record in _load_result_records(source_path):
            by_key.setdefault(_sample_key(record), record)
        records_by_file.append(by_key)

    shared_keys = set(records_by_file[0])
    for by_key in records_by_file[1:]:
        shared_keys &= set(by_key)

    if not shared_keys:
        raise ValueError("No shared sample keys found across selected result files")

    reference_records = records_by_file[0]
    by_attack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in shared_keys:
        record = reference_records[key]
        by_attack[str(record.get("attack_category", ""))].append(record)

    rng = random.Random(sample_seed)
    selected: list[dict[str, Any]] = []

    attack_categories = sorted(by_attack)
    base_quota = sample_size // len(attack_categories)
    remainder = sample_size % len(attack_categories)
    attack_order = list(attack_categories)
    rng.shuffle(attack_order)
    attack_quotas = {
        category: base_quota + (1 if idx < remainder else 0)
        for idx, category in enumerate(attack_order)
    }

    for attack_category in attack_categories:
        quota = attack_quotas[attack_category]
        attack_records = by_attack[attack_category]
        if len(attack_records) < quota:
            raise ValueError(
                f"Attack category {attack_category!r} has only {len(attack_records)} "
                f"shared records for quota {quota}"
            )

        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in attack_records:
            bucket_key = (
                str(record.get("level", "")),
                str(record.get("type", "")),
            )
            buckets[bucket_key].append(record)

        for bucket_records in buckets.values():
            bucket_records.sort(key=_sample_key)
            rng.shuffle(bucket_records)

        bucket_order = sorted(buckets)
        rng.shuffle(bucket_order)
        attack_selected: list[dict[str, Any]] = []

        while len(attack_selected) < quota and bucket_order:
            next_order: list[tuple[str, str]] = []
            for bucket_key in bucket_order:
                bucket_records = buckets[bucket_key]
                if not bucket_records:
                    continue
                attack_selected.append(bucket_records.pop())
                if len(attack_selected) == quota:
                    break
                if bucket_records:
                    next_order.append(bucket_key)
            bucket_order = next_order

        selected.extend(attack_selected)

    if len(selected) < sample_size:
        raise ValueError(
            f"Requested {sample_size} sample records but only {len(selected)} shared records exist"
        )

    selected.sort(key=_sample_key)
    return [_sample_file_entry(record) for record in selected]


def _load_or_create_sample_keys(
    sample_file: Path | None,
    *,
    result_files: list[tuple[str, str, Path]],
    sample_size: int | None,
    sample_seed: int,
    overwrite_sample: bool,
) -> set[str] | None:
    if sample_file is None:
        return None

    path = sample_file.expanduser().resolve()
    if path.exists() and not overwrite_sample:
        sample_keys = _load_sample_keys(path)
        logger.info("Loaded %d sample keys from %s", len(sample_keys), path)
        return sample_keys

    if sample_size is None:
        raise ValueError("--sample-size is required when creating a new --sample-file")

    entries = _build_balanced_sample(
        result_files,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    payload = {
        "sample_version": 1,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "balance_fields": ["level", "type", "attack_category"],
        "keys": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    sample_keys = {_normalize_sample_entry(entry) for entry in entries}
    logger.info("Created %d-key balanced sample at %s", len(sample_keys), path)
    return sample_keys


def _count_planned_calls(
    result_files: list[tuple[str, str, Path]],
    *,
    output_dir: Path,
    solver_slug: str,
    no_hint_cache: dict[str, dict[str, Any]],
    max_records: int | None,
    sample_keys: set[str] | None,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    pending_no_hint_keys = set(no_hint_cache)
    totals = {
        "source_records": 0,
        "already_done": 0,
        "new_no_hint_calls": 0,
        "new_hinted_calls": 0,
        "total_new_calls": 0,
        "input_tokens_est": 0.0,
    }

    for model_id, system_id, source_path in result_files:
        output_path = output_dir / model_id / f"{system_id}_hint_gain_{solver_slug}.jsonl"
        existing_by_source = (
            {}
            if overwrite
            else _load_jsonl_by_key(output_path, "hint_eval_source_key")
        )
        source_records = _filter_records_by_sample(
            _load_result_records(source_path),
            sample_keys,
        )
        if max_records is not None:
            source_records = source_records[:max_records]

        already_done = 0
        new_no_hint_calls = 0
        new_hinted_calls = 0
        input_tokens_est = 0.0
        for ordinal, _, record in source_records:
            if _source_key(model_id, system_id, ordinal) in existing_by_source:
                already_done += 1
                continue

            new_hinted_calls += 1
            input_tokens_est += _input_tokens_hinted(record)
            no_hint_key = _no_hint_key(record)
            if no_hint_key not in pending_no_hint_keys:
                pending_no_hint_keys.add(no_hint_key)
                new_no_hint_calls += 1
                input_tokens_est += _input_tokens_no_hint(record)

        rows.append(
            {
                "condition": f"{model_id}/{system_id}",
                "source_records": len(source_records),
                "already_done": already_done,
                "new_no_hint_calls": new_no_hint_calls,
                "new_hinted_calls": new_hinted_calls,
                "total_new_calls": new_no_hint_calls + new_hinted_calls,
                "input_tokens_est": input_tokens_est,
            }
        )
        for key in totals:
            totals[key] += rows[-1][key]

    return rows, totals


def print_estimate(
    rows: list[dict[str, Any]],
    totals: dict[str, int],
) -> None:
    header = (
        f"{'Condition':<25}  {'Rows':>6}  {'Done':>6}  "
        f"{'NoHint':>7}  {'Hinted':>7}  {'Calls':>7}  {'InMTok':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        input_mtok = row["input_tokens_est"] / 1_000_000
        print(
            f"{row['condition']:<25}  {row['source_records']:>6}  "
            f"{row['already_done']:>6}  {row['new_no_hint_calls']:>7}  "
            f"{row['new_hinted_calls']:>7}  {row['total_new_calls']:>7}  "
            f"{input_mtok:>7.3f}"
        )
    print("-" * len(header))
    total_input_mtok = totals["input_tokens_est"] / 1_000_000
    print(
        f"{'TOTAL':<25}  {totals['source_records']:>6}  "
        f"{totals['already_done']:>6}  {totals['new_no_hint_calls']:>7}  "
        f"{totals['new_hinted_calls']:>7}  {totals['total_new_calls']:>7}  "
        f"{total_input_mtok:>7.3f}"
    )
    print("=" * len(header))


def _find_result_files(
    results_dir: Path,
    models: list[str] | None,
    systems: list[str] | None,
) -> list[tuple[str, str, Path]]:
    files: list[tuple[str, str, Path]] = []
    model_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    if models:
        model_dirs = [path for path in model_dirs if path.name in models]

    for model_dir in model_dirs:
        model_id = model_dir.name
        if model_id.startswith("_"):
            continue
        for path in sorted(model_dir.glob("*.jsonl")):
            system_id = path.stem
            if system_id.endswith("_evaluated") or "_hint_gain_" in system_id:
                continue
            if systems and system_id not in systems:
                continue
            files.append((model_id, system_id, path))

    return files


def _resolve_models_and_systems(
    args: argparse.Namespace,
    config_data: dict[str, object],
) -> tuple[list[str] | None, list[str] | None]:
    models = config_data.get("models")
    systems = config_data.get("systems")

    if args.models is not None:
        models = args.models
    if args.systems is not None:
        systems = args.systems

    return (
        list(models) if isinstance(models, list) else None,
        list(systems) if isinstance(systems, list) else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate solver improvement from hints.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config file. CLI flags override config values.",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--systems", nargs="+", default=None, choices=ALL_SYSTEMS)
    parser.add_argument("--solver-model-name", default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=None,
        help=(
            "JSON file containing selected hint-eval keys. If it does not exist, "
            "--sample-size is used to create a balanced sample."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of shared records to sample when creating --sample-file.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=13,
        help="Random seed used only when creating a new --sample-file.",
    )
    parser.add_argument(
        "--overwrite-sample",
        action="store_true",
        help="Regenerate --sample-file instead of loading it when it already exists.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional per-file cap for pilot runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate per-condition hint-gain files. The no-hint cache is retained.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print planned solver-call counts and exit without loading a model.",
    )
    args = parser.parse_args()

    try:
        config_data, config_base = load_config(args.config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    results_dir = (
        resolve_config_path(config_data["results_dir"], config_base)
        if "results_dir" in config_data
        else args.results_dir.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else results_dir
    models, systems = _resolve_models_and_systems(args, config_data)

    if not results_dir.exists():
        parser.error(f"results directory does not exist: {results_dir}")

    solver_slug = f"{_solver_slug(args.solver_model_name)}_{PROMPT_VERSION}"
    no_hint_cache_path = output_dir / f"hint_gain_no_hint_cache_{solver_slug}.jsonl"
    no_hint_cache = _load_jsonl_by_key(no_hint_cache_path, "hint_eval_no_hint_key")

    result_files = _find_result_files(results_dir, models, systems)
    if not result_files:
        logger.warning("No result files found under %s.", results_dir)
        return

    try:
        sample_keys = _load_or_create_sample_keys(
            args.sample_file,
            result_files=result_files,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            overwrite_sample=args.overwrite_sample,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    if args.estimate_only:
        rows, totals = _count_planned_calls(
            result_files,
            output_dir=output_dir,
            solver_slug=solver_slug,
            no_hint_cache=no_hint_cache,
            max_records=args.max_records,
            sample_keys=sample_keys,
            overwrite=args.overwrite,
        )
        print_estimate(rows, totals)
        return

    solver = _load_solver(args)

    all_aggs: dict[str, dict[str, Any]] = {}
    logger.info("Evaluating %d result file(s).", len(result_files))

    for model_id, system_id, source_path in result_files:
        condition = f"{model_id}/{system_id}"
        output_path = output_dir / model_id / f"{system_id}_hint_gain_{solver_slug}.jsonl"

        if args.overwrite and output_path.exists():
            output_path.unlink()

        existing_by_source = (
            {}
            if args.overwrite
            else _load_jsonl_by_key(output_path, "hint_eval_source_key")
        )

        source_records = _filter_records_by_sample(
            _load_result_records(source_path),
            sample_keys,
        )
        if args.max_records is not None:
            source_records = source_records[: args.max_records]

        condition_records: list[dict[str, Any]] = []
        for ordinal, line_no, record in tqdm(source_records, desc=condition, unit="record"):
            source_key = _source_key(model_id, system_id, ordinal)
            if source_key in existing_by_source:
                condition_records.append(existing_by_source[source_key])
                continue

            case_key = _case_key(record)
            no_hint_key = _no_hint_key(record)
            no_hint_entry = no_hint_cache.get(no_hint_key)
            if no_hint_entry is None:
                no_hint_output = solver.generate(
                    SOLVER_SYSTEM,
                    _no_hint_prompt(record),
                    max_tokens=args.max_tokens,
                )
                no_hint_check = _check_solver_output(
                    no_hint_output,
                    str(record.get("gold_answer", "")),
                )
                no_hint_entry = {
                    "hint_eval_no_hint_key": no_hint_key,
                    "hint_eval_case_key": case_key,
                    "prompt_version": PROMPT_VERSION,
                    "solver_backend": SOLVER_BACKEND,
                    "solver_model_name": args.solver_model_name,
                    "question_uid": record.get("question_uid", ""),
                    "question_idx": record.get("question_idx"),
                    "attack_id": record.get("attack_id"),
                    "attack_category": record.get("attack_category", ""),
                    "gold_answer": record.get("gold_answer", ""),
                    "no_hint_solver_output": no_hint_output,
                    "eval_no_hint_solve_correct": no_hint_check["correct"],
                    "eval_no_hint_extracted_answer": no_hint_check["extracted_answer"],
                    "eval_gold_answer_normalized": no_hint_check["gold_answer_normalized"],
                }
                _append_jsonl(no_hint_cache_path, no_hint_entry)
                no_hint_cache[no_hint_key] = no_hint_entry

            hinted_output = solver.generate(
                SOLVER_SYSTEM,
                _hinted_prompt(record),
                max_tokens=args.max_tokens,
            )
            hinted_check = _check_solver_output(
                hinted_output,
                str(record.get("gold_answer", "")),
            )

            no_hint_correct = bool(no_hint_entry["eval_no_hint_solve_correct"])
            hinted_correct = bool(hinted_check["correct"])

            annotated = dict(record)
            annotated.update(
                {
                    "hint_eval_source_key": source_key,
                    "hint_eval_sample_key": _sample_key(record),
                    "hint_eval_source_line": line_no,
                    "hint_eval_source_ordinal": ordinal,
                    "hint_eval_case_key": case_key,
                    "hint_eval_no_hint_key": no_hint_key,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_includes_student_query": False,
                    "source_model": model_id,
                    "source_system": system_id,
                    "solver_backend": SOLVER_BACKEND,
                    "solver_model_name": args.solver_model_name,
                    "no_hint_solver_output": no_hint_entry["no_hint_solver_output"],
                    "hinted_solver_output": hinted_output,
                    "eval_no_hint_solve_correct": no_hint_correct,
                    "eval_hinted_solve_correct": hinted_correct,
                    "eval_no_hint_extracted_answer": no_hint_entry[
                        "eval_no_hint_extracted_answer"
                    ],
                    "eval_hinted_extracted_answer": hinted_check["extracted_answer"],
                    "eval_gold_answer_normalized": hinted_check[
                        "gold_answer_normalized"
                    ],
                    "eval_hint_gain_outcome": _outcome(no_hint_correct, hinted_correct),
                }
            )
            _append_jsonl(output_path, annotated)
            condition_records.append(annotated)

        all_aggs[condition] = aggregate(condition_records)
        logger.info("Saved %s", output_path)

    print_summary_table(all_aggs)

    summary_csv_path = output_dir / f"hint_gain_summary_{solver_slug}.csv"
    save_summary_csv(all_aggs, summary_csv_path)
    logger.info("Summary CSV saved to %s", summary_csv_path)

    summary_json_path = output_dir / f"hint_gain_summary_{solver_slug}.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_aggs, f, indent=2, ensure_ascii=False)
    logger.info("Summary JSON saved to %s", summary_json_path)


if __name__ == "__main__":
    main()
