# utils.py -- Shared config, paths, constants, and JSONL reading.
#
# Other modules import from here to get project paths, model settings,
# experiment defaults, and the JSONL line reader.

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _PROJECT_ROOT
load_dotenv(_PROJECT_ROOT / ".env")

# Paths
DATA_DIR = _PROJECT_ROOT / "Data"
RESULTS_DIR = _PROJECT_ROOT / "results"
# Generated locally from the official MATH benchmark download.
MATH_DATASET_PATH = DATA_DIR / "math.json"
ATTACK_PROMPTS_PATH = DATA_DIR / "dataset_b.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# API keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# Experiment constants
SAMPLE_SIZE: int = 500          # Number of math questions to sample
NUM_HINT_STEPS: int = 5         # Default N for hint generation
RANDOM_SEED: int = 42

# System identifiers
SystemId = Literal[
    "B0",
    "B1",
    "TS-Weak",
    "TS-Medium",
    "TS-Strict",
    "SS-Medium",
    "SS-Strict",
]
ALL_SYSTEMS: list[SystemId] = [
    "B0",
    "B1",
    "TS-Weak",
    "TS-Medium",
    "TS-Strict",
    "SS-Medium",
    "SS-Strict",
]

# Model identifiers
ModelId = Literal["llama", "general", "math"]
ALL_MODELS: list[ModelId] = ["llama", "general", "math"]


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_id: ModelId
    backend: Literal["openai", "huggingface"]
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.95


# Default model configurations
MODELS: dict[ModelId, ModelConfig] = {
    "llama": ModelConfig(
        model_id="llama",
        backend="huggingface",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.3,
        max_tokens=2048,
    ),
    "general": ModelConfig(
        model_id="general",
        backend="openai",
        model_name="gpt-4o-mini",
        temperature=0.3,
        max_tokens=2048,
    ),
    "math": ModelConfig(
        model_id="math",
        backend="huggingface",
        model_name="Qwen/Qwen2.5-Math-7B-Instruct",
        temperature=0.3,
        max_tokens=2048,
    ),
}


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    sample_size: int = SAMPLE_SIZE
    num_hint_steps: int = NUM_HINT_STEPS
    seed: int = RANDOM_SEED
    systems: list[SystemId] = field(default_factory=lambda: list(ALL_SYSTEMS))
    models: list[ModelId] = field(default_factory=lambda: list(ALL_MODELS))
    results_dir: Path = RESULTS_DIR
    resume_source_dirs: list[Path] = field(default_factory=list)
    resume: bool = True   # skip already-generated outputs on restart
    batch_size: int = 8   # GPU batch size for HuggingFace models


# --- JSONL helpers ---

def iter_jsonl_objects(
    path: Path,
    *,
    logger: logging.Logger | None = None,
    label: str | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (line_number, dict) pairs from a JSONL file.

    Handles corrupted lines: skips bad JSON and splits lines where
    multiple JSON objects got glued together (from interrupted writes).
    """
    decoder = json.JSONDecoder()
    source_label = label or str(path)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            text = raw_line.strip()
            if not text:
                continue

            idx = 0
            warned_about_concat = False
            decoded_count = 0

            while idx < len(text):
                while idx < len(text) and text[idx].isspace():
                    idx += 1
                if idx >= len(text):
                    break

                try:
                    rec, end = decoder.raw_decode(text, idx)
                except json.JSONDecodeError as exc:
                    next_idx = text.find("{", idx + 1)
                    while next_idx != -1:
                        try:
                            rec, end = decoder.raw_decode(text, next_idx)
                        except json.JSONDecodeError:
                            next_idx = text.find("{", next_idx + 1)
                            continue

                        if not isinstance(rec, dict):
                            next_idx = text.find("{", next_idx + 1)
                            continue

                        if logger is not None:
                            logger.warning(
                                "Recovered JSON object after malformed content in %s at line %d",
                                source_label,
                                line_no,
                            )
                        idx = next_idx
                        break
                    else:
                        if logger is not None:
                            logger.warning(
                                "Skipping malformed JSON in %s at line %d column %d: %s",
                                source_label,
                                line_no,
                                exc.colno,
                                exc.msg,
                            )
                        break

                if not isinstance(rec, dict):
                    if logger is not None:
                        logger.warning(
                            "Skipping non-object JSON in %s at line %d",
                            source_label,
                            line_no,
                        )
                    idx = end
                    continue

                if decoded_count > 0 and logger is not None and not warned_about_concat:
                    logger.warning(
                        "Recovered concatenated JSON objects in %s at line %d",
                        source_label,
                        line_no,
                    )
                    warned_about_concat = True

                yield line_no, rec
                decoded_count += 1
                idx = end


# --- Config file helpers ---

def load_config(config_path: Path | None) -> tuple[dict[str, object], Path]:
    """Read a JSON config file. Returns (parsed dict, folder the file is in)."""
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


def resolve_config_path(value: str | Path, base_dir: Path) -> Path:
    """Turn a relative path from a config file into an absolute path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in (".", ".."):
        return (base_dir / path).resolve()
    return (PROJECT_ROOT / path).resolve()
