"""
Central configuration for the Two-Step Prompting Tutor Wrapper experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
SystemId = Literal["B0", "B1", "TS-Weak", "TS-Medium", "TS-Strict"]
ALL_SYSTEMS: list[SystemId] = ["B0", "B1", "TS-Weak", "TS-Medium", "TS-Strict"]

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
    resume: bool = True   # skip already-generated outputs on restart
    batch_size: int = 8   # GPU batch size for HuggingFace models
