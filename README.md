# Tutor

This project evaluates how well LLM tutors avoid giving final math answers too early.
It runs experiments, then computes leakage and compliance metrics.

## Quick Start

From the project root:

```bash
./bin/install
./bin/run
```

That is the full workflow.

## What These Commands Do

- `./bin/install`
  - creates `.venv`
  - installs dependencies
  - builds `Data/math.json` (500-question local dataset)

- `./bin/run`
  - runs `scripts/run_experiment.py`
  - runs `scripts/evaluate_results.py`
  - uses `config/experiment.json` by default

## Output Files

Main outputs are in the `results_dir` from `config/experiment.json` (default: `results/`):

- `results/<model>/<system>.jsonl`
- `results/<model>/<system>_evaluated.jsonl`
- `results/summary.csv`
- `results/summary.json`

## Config

Main config file:

- `config/experiment.json`

Common fields you may change:

- `sample_size`
- `models`
- `systems`
- `results_dir`

You can also run with another config path:

```bash
./bin/run /path/to/your_config.json
```

## Dataset Note

This repo does not store the full third-party MATH benchmark file.
Dataset preparation is done locally by `scripts/prepare_math_dataset.py`, which downloads from the official source and creates a deterministic 500-question subset.

If you need to rebuild it manually:

```bash
python scripts/prepare_math_dataset.py
```

## Adversarial Prompts

The adversarial prompts in `Data/dataset_b.json` were hand-written to reflect realistic student pressure tactics in tutoring chats (for example: direct answer requests, exam-time urgency, yes/no confirmation, and instruction override attempts). We used these prompts to test whether the tutor policy still avoids final-answer leakage under plausible user behavior, not only under cooperative prompts.

## Project Files

- `scripts/run_experiment.py`
- `scripts/evaluate_results.py`
- `scripts/evaluate_hint_gain.py`
- `scripts/paired_significance.py`
- `scripts/prepare_math_dataset.py`
- `src/pipeline.py`
- `src/evaluation.py`
- `src/config.py`
- `src/utils.py`
- `config/experiment.json`
- `config/ss_general_pilot.json`
- `config/hint_gain_phi100_sample.json`
