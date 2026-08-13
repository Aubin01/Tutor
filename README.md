# Tutor

Tutor contains the code and experimental pipeline for Strong Solvers, Leaky Tutors: Evaluating Answer Leakage in LLM Math Tutors. The project evaluates whether large language models can use mathematical solution context to generate pedagogical hints while withholding final answers.

## Quick Start

From the project root:

```bash
./bin/install
./bin/run
```
## What These Commands Do

- `./bin/install`
  - creates `.venv`
  - installs dependencies
  - builds `Data/math.json` (500-problem local dataset)

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
Dataset preparation is done locally by `scripts/prepare_math_dataset.py`, which downloads from the official source and creates a deterministic 500-problem subset.

If you need to rebuild it manually:

```bash
python scripts/prepare_math_dataset.py
```

## Student Prompts

The student prompts in `Data/dataset_b.json` were hand-written to reflect realistic student pressure tactics in tutoring chats (for example: direct answer requests, exam-time urgency, yes/no confirmation, and instruction override attempts). We used these prompts to test whether the tutor policy still avoids final-answer leakage under plausible user behavior, not only under cooperative prompts.

## Paper: Strong Solvers, Leaky Tutors: Evaluating Answer Leakage in LLM Math Tutors

Aubin Mugisha and Behrooz Mansouri<br>
University of Southern Maine
