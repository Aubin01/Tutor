"""Main experiment script.

Runs each model on each system for every question and attack prompt.
Saves results as JSONL files under the results directory.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --systems B0 B1
    python scripts/run_experiment.py --models general
    python scripts/run_experiment.py --config config/experiment.json
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import logging
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Silence noisy HuggingFace / torch warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

import transformers
transformers.logging.set_verbosity_error()

from tqdm import tqdm

# --- Make project importable from repo root ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    ALL_MODELS,
    ALL_SYSTEMS,
    ATTACK_PROMPTS_PATH,
    MATH_DATASET_PATH,
    MODELS,
    RESULTS_DIR,
    ExperimentConfig,
    ModelConfig,
    iter_jsonl_objects,
    load_config,
    resolve_config_path,
    PROJECT_ROOT,
)
from src.pipeline import (
    build_test_cases,
    load_attack_prompts,
    load_math_questions,
    sample_questions,
    load_model,
    run_system_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy HTTP request logs from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_MATH_DATASET_PATH = MATH_DATASET_PATH
DEFAULT_ATTACK_PROMPTS_PATH = ATTACK_PROMPTS_PATH
STORAGE_ERRNOS = tuple(
    code for code in (errno.ENOSPC, getattr(errno, "EDQUOT", None)) if code is not None
)


class StorageExhaustedError(RuntimeError):
    """Raised when result files cannot be persisted because storage is exhausted."""


# --- I/O helpers ---

def _output_path(results_dir: Path, model_id: str, system_id: str) -> Path:
    """results/<model>/<system>.jsonl"""
    d = results_dir / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{system_id}.jsonl"


def _legacy_case_key(question_idx: Any, attack_id: Any) -> str:
    return f"idx:{question_idx}_{attack_id}"


def _make_case_key(question_uid: str | None, question_idx: Any, attack_id: Any) -> str:
    """Primary stable key for resume logic."""
    if question_uid:
        return f"{question_uid}_{attack_id}"
    return _legacy_case_key(question_idx, attack_id)


def _candidate_case_keys(question_uid: str | None, question_idx: Any, attack_id: Any) -> set[str]:
    """All keys that can represent the same case (new + legacy)."""
    keys = {_legacy_case_key(question_idx, attack_id)}
    if question_uid:
        keys.add(f"{question_uid}_{attack_id}")
    return keys


def _step_a_case_key(case: dict[str, Any]) -> str:
    """Stable per-question key for the hidden Step A precompute phase."""
    return str(case.get("question_uid") or case["question_idx"])


def _step_a_cache_path(results_dir: Path, model_id: str) -> Path:
    cache_dir = results_dir / "_step_a_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model_id}.jsonl"


def _load_existing_keys(path: Path) -> set[str]:
    """Return case keys already saved in this results file (for resume)."""
    keys: set[str] = set()
    if path.exists():
        for _, rec in iter_jsonl_objects(path, logger=logger, label=str(path)):
            attack_id = rec.get("attack_id")
            if attack_id is None:
                continue

            q_idx = rec.get("question_idx")
            q_uid = rec.get("question_uid")
            keys |= _candidate_case_keys(q_uid, q_idx, attack_id)
    return keys


def _format_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{int(num_bytes)}B"


def _free_bytes(path: Path) -> int | None:
    target = path if path.is_dir() else path.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        return shutil.disk_usage(target).free
    except OSError:
        return None


def _raise_if_storage_exhausted(path: Path, exc: OSError) -> None:
    if exc.errno not in STORAGE_ERRNOS:
        return

    free_bytes = _free_bytes(path)
    detail = f"Storage exhausted while writing {path}: {exc.strerror or exc}"
    if free_bytes is not None:
        detail += f" (filesystem currently reports {_format_bytes(free_bytes)} free)."
        if free_bytes > 0:
            detail += " This usually means the device filled earlier during the run or a quota limit was hit."
    raise StorageExhaustedError(detail) from exc


def _append_result(path: Path, result: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except OSError as exc:
        _raise_if_storage_exhausted(path, exc)
        raise


def _load_step_a_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not path.exists():
        return cache

    for _, rec in iter_jsonl_objects(path, logger=logger, label=str(path)):
        key = rec.get("question_key")
        step_a_output = rec.get("step_a_output")
        if key and step_a_output:
            cache[str(key)] = str(step_a_output)

    return cache


def _append_step_a_cache(path: Path, question_key: str, step_a_output: str) -> None:
    rec = {
        "question_key": question_key,
        "step_a_output": step_a_output,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as exc:
        _raise_if_storage_exhausted(path, exc)
        raise


def _recover_step_a_cache(
    cache: dict[str, str],
    result_paths: list[Path],
) -> dict[str, str]:
    recovered: dict[str, str] = {}
    seen_paths: set[Path] = set()

    for result_path in result_paths:
        try:
            resolved_path = result_path.resolve()
        except OSError:
            resolved_path = result_path
        if resolved_path in seen_paths or not _path_exists(result_path):
            continue
        seen_paths.add(resolved_path)

        for _, rec in iter_jsonl_objects(result_path, logger=logger, label=str(result_path)):
            step_a_output = rec.get("step_a_output")
            if not step_a_output:
                continue

            question_key = str(rec.get("question_uid") or rec.get("question_idx") or "")
            if not question_key or question_key in cache or question_key in recovered:
                continue

            recovered[question_key] = str(step_a_output)

    cache.update(recovered)
    return recovered


def _seed_resume_file(path: Path, source_paths: list[Path], label: str) -> None:
    should_seed = not path.exists()
    if path.exists():
        try:
            should_seed = path.stat().st_size == 0
        except OSError:
            should_seed = False

    if not should_seed:
        return

    target_resolved = _safe_resolve(path)
    for source_path in source_paths:
        source_resolved = _safe_resolve(source_path)
        if source_resolved == target_resolved:
            continue
        if not _path_exists(source_path):
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Seeding %s from %s", label, source_path)
        try:
            shutil.copyfile(source_path, path)
        except OSError as exc:
            _raise_if_storage_exhausted(path, exc)
            raise
        return


def _export_hints_csv(jsonl_path: Path) -> Path:
    """Write a simple CSV (question_id, attack_id, hint) from a JSONL results file."""
    csv_path = jsonl_path.with_suffix(".csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=["question_id", "attack_id", "hint"])
            writer.writeheader()
            for _, rec in iter_jsonl_objects(jsonl_path, logger=logger, label=str(jsonl_path)):
                writer.writerow({
                    "question_id": rec.get("question_idx", ""),
                    "attack_id": rec.get("attack_id", ""),
                    "hint": rec.get("output", ""),
                })
        return csv_path
    except OSError as exc:
        _raise_if_storage_exhausted(csv_path, exc)
        raise


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser()


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        logger.warning("Skipping inaccessible path %s: %s", path, exc)
        return False


# --- Main ---

def run(
    cfg: ExperimentConfig,
    math_dataset_path: Path,
    attack_prompts_path: Path,
):
    current_results_dir = _safe_resolve(cfg.results_dir)
    resume_source_dirs: list[Path] = []
    seen_resume_sources: set[Path] = set()
    for candidate in [*cfg.resume_source_dirs, RESULTS_DIR]:
        resolved = _safe_resolve(candidate)
        if resolved == current_results_dir or resolved in seen_resume_sources:
            continue
        resume_source_dirs.append(resolved)
        seen_resume_sources.add(resolved)

    # --- Load data ---────
    logger.info("Loading datasets: %s and %s", math_dataset_path, attack_prompts_path)
    all_questions = load_math_questions(math_dataset_path)
    questions = sample_questions(all_questions, cfg.sample_size, cfg.seed)
    attack_prompts = load_attack_prompts(attack_prompts_path)
    test_cases = build_test_cases(questions, attack_prompts)
    logger.info(
        "Prepared %d test cases (%d questions × %d attack prompts)",
        len(test_cases), len(questions), len(attack_prompts),
    )

    # --- Run each (model, system) condition ---
    for model_id in cfg.models:
        model_cfg: ModelConfig = MODELS[model_id]
        logger.info("Loading model: %s (%s)", model_cfg.model_name, model_cfg.backend)
        model = load_model(model_cfg)

        # Step A cache: avoids re-solving the same question for each
        # TS variant. Saved to disk so restarts skip finished work.
        step_a_cache_file = _step_a_cache_path(cfg.results_dir, model_id)
        if cfg.resume and resume_source_dirs:
            _seed_resume_file(
                step_a_cache_file,
                [src / "_step_a_cache" / f"{model_id}.jsonl" for src in resume_source_dirs],
                f"Step A cache for {model_id}",
            )
        if not cfg.resume and step_a_cache_file.exists():
            step_a_cache_file.unlink()
        step_a_cache = _load_step_a_cache(step_a_cache_file) if cfg.resume else {}
        if cfg.resume:
            recovered_step_a = _recover_step_a_cache(
                step_a_cache,
                [
                    root / model_id / f"{system_id}.jsonl"
                    for root in [cfg.results_dir, *resume_source_dirs]
                    for system_id in ("TS-Weak", "TS-Medium", "TS-Strict")
                ],
            )
            if recovered_step_a:
                logger.info(
                    "Recovered %d Step A answers for %s from existing TS result files",
                    len(recovered_step_a), model_id,
                )
                for question_key, step_a_output in recovered_step_a.items():
                    _append_step_a_cache(step_a_cache_file, question_key, step_a_output)
        if step_a_cache:
            logger.info(
                "Loaded %d persisted Step A answers for %s from %s",
                len(step_a_cache), model_id, step_a_cache_file.name,
            )

        for system_id in cfg.systems:
            out_path = _output_path(cfg.results_dir, model_id, system_id)
            if cfg.resume and resume_source_dirs:
                _seed_resume_file(
                    out_path,
                    [src / model_id / f"{system_id}.jsonl" for src in resume_source_dirs],
                    f"results for {model_id}/{system_id}",
                )
            if not cfg.resume and out_path.exists():
                out_path.unlink()
            done_keys = _load_existing_keys(out_path) if cfg.resume else set()
            logger.info(
                "  System %-10s → %s  (%d already done)",
                system_id, out_path.name, len(done_keys),
            )

            total = len(test_cases)
            to_run = []
            for c in test_cases:
                c_keys = _candidate_case_keys(
                    c.get("question_uid"),
                    c.get("question_idx"),
                    c.get("attack_id"),
                )
                if done_keys.isdisjoint(c_keys):
                    to_run.append(c)
            if len(to_run) < total:
                logger.info(
                    "    Skipping %d already done, %d remaining.",
                    total - len(to_run), len(to_run)
                )

            errors = 0
            step_a_pbar = None
            pbar_position = 0
            if system_id in ("TS-Weak", "TS-Medium", "TS-Strict"):
                pending_step_a_keys = {
                    _step_a_case_key(c)
                    for c in to_run
                    if _step_a_case_key(c) not in step_a_cache
                }
                if pending_step_a_keys:
                    step_a_pbar = tqdm(
                        total=len(pending_step_a_keys),
                        desc=f"  {model_id}/{system_id} Step A",
                        unit="question",
                        position=0,
                        bar_format="{desc}  {bar} {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
                    )
                    pbar_position = 1
            pbar = tqdm(
                total=len(to_run),
                desc=f"  {model_id}/{system_id}",
                unit="case",
                position=pbar_position,
                bar_format="{desc}  {bar} {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
            )
            try:
                for result in run_system_batch(
                    system_id=system_id,
                    model=model,
                    cases=to_run,
                    n_steps=cfg.num_hint_steps,
                    step_a_cache=step_a_cache,
                    batch_size=cfg.batch_size,
                    step_a_result_callback=(
                        lambda question_key, step_a_output, path=step_a_cache_file:
                        _append_step_a_cache(path, question_key, step_a_output)
                    ),
                    step_a_progress_callback=step_a_pbar.update if step_a_pbar else None,
                ):
                    try:
                        result["model"] = model_id
                        _append_result(out_path, result)
                        pbar.update(1)
                    except StorageExhaustedError:
                        logger.error(
                            "Stopping run after storage failure while saving q=%s, atk=%s",
                            result.get("question_idx"), result.get("attack_id"),
                        )
                        raise
                    except Exception:
                        logger.exception(
                            "Error saving result q=%s, atk=%s",
                            result.get("question_idx"), result.get("attack_id"),
                        )
                        errors += 1
            finally:
                pbar.close()
                if step_a_pbar is not None:
                    step_a_pbar.close()

            logger.info(
                "  ✓ %s/%s  done  (%d generated, %d errors)",
                model_id, system_id, len(to_run) - errors, errors,
            )

            # Export a clean CSV of hints for human evaluation
            csv_path = _export_hints_csv(out_path)
            logger.info("  → Hints CSV saved: %s", csv_path)

    logger.info("All conditions complete. Results in %s", cfg.results_dir)


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Run the Two-Step Prompting Tutor experiment."
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON config file. CLI flags override config values.",
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        choices=ALL_SYSTEMS,
        help="Systems to test (default: all).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=ALL_MODELS,
        help="Models to use (default: all). Options: llama, general, math.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of math questions to sample (default: 500).",
    )
    parser.add_argument(
        "--hint-steps", type=int, default=None,
        help="Number of hint steps N (default: 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Legacy sampling seed (kept for compatibility; deterministic sampler ignores it).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not resume from existing outputs.",
    )
    parser.add_argument(
        "--use-local-hf", action="store_true",
        help="Load HuggingFace models locally (default behaviour; flag accepted for compatibility).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="GPU batch size for HuggingFace models (default: 8). Lower if you get OOM.",
    )
    parser.add_argument(
        "--math-dataset", type=Path, default=DEFAULT_MATH_DATASET_PATH,
        help="Path to the generated math dataset JSON (default: Data/math.json).",
    )
    parser.add_argument(
        "--attack-prompts", type=Path, default=DEFAULT_ATTACK_PROMPTS_PATH,
        help="Path to the attack prompts JSON (default: Data/dataset_b.json).",
    )
    args = parser.parse_args()

    try:
        config_data, config_base = load_config(args.config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    cfg = ExperimentConfig()
    if config_data.get("systems") is not None:
        cfg.systems = list(config_data["systems"])
    if config_data.get("models") is not None:
        cfg.models = list(config_data["models"])
    cfg.sample_size = int(config_data.get("sample_size", cfg.sample_size))
    cfg.num_hint_steps = int(config_data.get("num_hint_steps", cfg.num_hint_steps))
    cfg.seed = int(config_data.get("seed", cfg.seed))
    cfg.resume = bool(config_data.get("resume", cfg.resume))
    cfg.batch_size = int(config_data.get("batch_size", cfg.batch_size))
    if "results_dir" in config_data:
        cfg.results_dir = resolve_config_path(config_data["results_dir"], config_base)
    if config_data.get("resume_source_dirs") is not None:
        cfg.resume_source_dirs = [
            resolve_config_path(path, config_base)
            for path in config_data["resume_source_dirs"]
        ]

    if args.systems:
        cfg.systems = args.systems
    if args.models:
        cfg.models = args.models
    if args.sample_size:
        cfg.sample_size = args.sample_size
    if args.hint_steps:
        cfg.num_hint_steps = args.hint_steps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_resume:
        cfg.resume = False
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    math_dataset_path = resolve_config_path(
        config_data.get("math_dataset_path", DEFAULT_MATH_DATASET_PATH),
        config_base,
    )
    if args.math_dataset != DEFAULT_MATH_DATASET_PATH:
        math_dataset_path = args.math_dataset.expanduser().resolve()

    attack_prompts_path = resolve_config_path(
        config_data.get("attack_prompts_path", DEFAULT_ATTACK_PROMPTS_PATH),
        config_base,
    )
    if args.attack_prompts != DEFAULT_ATTACK_PROMPTS_PATH:
        attack_prompts_path = args.attack_prompts.expanduser().resolve()

    for label, path in (
        ("math dataset", math_dataset_path),
        ("attack prompts dataset", attack_prompts_path),
    ):
        if not path.exists():
            parser.error(f"{label} not found: {path}")
        if not path.is_file():
            parser.error(f"{label} is not a file: {path}")

    try:
        run(cfg, math_dataset_path=math_dataset_path, attack_prompts_path=attack_prompts_path)
    except StorageExhaustedError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
