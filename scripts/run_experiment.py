#!/usr/bin/env python3
"""
run_experiment.py — Main experiment runner.

Generates outputs for every (model × system × question × attack_prompt)
combination and saves results as JSONL files under results/.

Usage
-----
    python scripts/run_experiment.py                       # run all conditions
    python scripts/run_experiment.py --systems B0 B1       # run only baselines
    python scripts/run_experiment.py --models general       # one model only
    python scripts/run_experiment.py --sample-size 50       # quick test run
    python scripts/run_experiment.py --config config/experiment.json
    python scripts/run_experiment.py --math-dataset /path/to/math.json \
        --attack-prompts /path/to/dataset_b.json
    python scripts/run_experiment.py --use-local-hf         # load HF model locally
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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

# ── Make project importable from repo root ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ALL_MODELS,
    ALL_SYSTEMS,
    ATTACK_PROMPTS_PATH,
    MATH_DATASET_PATH,
    MODELS,
    ExperimentConfig,
    ModelConfig,
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATH_DATASET_PATH = MATH_DATASET_PATH
DEFAULT_ATTACK_PROMPTS_PATH = ATTACK_PROMPTS_PATH


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

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


def _load_existing_keys(path: Path) -> set[str]:
    """Return set of stable + legacy case keys already generated."""
    keys: set[str] = set()
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    attack_id = rec.get("attack_id")
                    if attack_id is None:
                        continue

                    q_idx = rec.get("question_idx")
                    q_uid = rec.get("question_uid")

                    keys |= _candidate_case_keys(q_uid, q_idx, attack_id)
                except json.JSONDecodeError:
                    continue
    return keys


def _append_result(path: Path, result: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _export_hints_csv(jsonl_path: Path) -> Path:
    """Read a JSONL results file and write a simple CSV with question_id and hint."""
    csv_path = jsonl_path.with_suffix(".csv")
    rows: list[dict] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            rows.append({
                "question_id": rec.get("question_idx", ""),
                "attack_id": rec.get("attack_id", ""),
                "hint": rec.get("output", ""),
            })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "attack_id", "hint"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _load_config(config_path: Path | None) -> tuple[dict[str, Any], Path]:
    if config_path is None:
        return {}, PROJECT_ROOT

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
    # (e.g., "Data/math.json" -> <repo>/Data/math.json).
    # Use "./" or "../" in config when you explicitly want config-file-relative paths.
    if path.parts and path.parts[0] in (".", ".."):
        return (base_dir / path).resolve()
    return (PROJECT_ROOT / path).resolve()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    cfg: ExperimentConfig,
    math_dataset_path: Path,
    attack_prompts_path: Path,
):
    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading datasets: %s and %s", math_dataset_path, attack_prompts_path)
    all_questions = load_math_questions(math_dataset_path)
    questions = sample_questions(all_questions, cfg.sample_size, cfg.seed)
    attack_prompts = load_attack_prompts(attack_prompts_path)
    test_cases = build_test_cases(questions, attack_prompts)
    logger.info(
        "Prepared %d test cases (%d questions × %d attack prompts)",
        len(test_cases), len(questions), len(attack_prompts),
    )

    # ── Run each (model, system) condition ────────────────────────────────
    for model_id in cfg.models:
        model_cfg: ModelConfig = MODELS[model_id]
        logger.info("Loading model: %s (%s)", model_cfg.model_name, model_cfg.backend)
        model = load_model(model_cfg)

        # Step A cache: avoids re-solving the same question for each
        # strictness level / attack prompt on the same model.
        step_a_cache: dict[str, str] = {}

        for system_id in cfg.systems:
            out_path = _output_path(cfg.results_dir, model_id, system_id)
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
            pbar = tqdm(
                total=len(to_run),
                desc=f"  {model_id}/{system_id}",
                unit="case",
                bar_format="{desc}  {bar} {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
            )
            for result in run_system_batch(
                system_id=system_id,
                model=model,
                cases=to_run,
                n_steps=cfg.num_hint_steps,
                step_a_cache=step_a_cache,
                batch_size=cfg.batch_size,
            ):
                try:
                    result["model"] = model_id
                    _append_result(out_path, result)
                    pbar.update(1)
                except Exception:
                    logger.exception(
                        "Error saving result q=%s, atk=%s",
                        result.get("question_idx"), result.get("attack_id"),
                    )
                    errors += 1
            pbar.close()

            logger.info(
                "  ✓ %s/%s  done  (%d generated, %d errors)",
                model_id, system_id, len(to_run) - errors, errors,
            )

            # Export a clean CSV of hints for human evaluation
            csv_path = _export_hints_csv(out_path)
            logger.info("  → Hints CSV saved: %s", csv_path)

    logger.info("All conditions complete. Results in %s", cfg.results_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

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
        config_data, config_base = _load_config(args.config)
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
        cfg.results_dir = _resolve_config_path(config_data["results_dir"], config_base)

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

    math_dataset_path = _resolve_config_path(
        config_data.get("math_dataset_path", DEFAULT_MATH_DATASET_PATH),
        config_base,
    )
    if args.math_dataset != DEFAULT_MATH_DATASET_PATH:
        math_dataset_path = args.math_dataset.expanduser().resolve()

    attack_prompts_path = _resolve_config_path(
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

    run(cfg, math_dataset_path=math_dataset_path, attack_prompts_path=attack_prompts_path)


if __name__ == "__main__":
    main()
