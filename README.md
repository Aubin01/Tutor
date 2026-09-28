# Tutor

Tutor contains the code and experimental pipeline for **Strong Solvers, Leaky Tutors: Evaluating Answer Leakage in LLM Math Tutors**. The project evaluates whether large language models can use mathematical solution context to generate pedagogical hints while withholding final answers.

The paper was accepted at the 2026 ACM/IEEE Joint Conference on Digital Libraries (JCDL '26). [Read the accepted paper](Strong_Solvers_Leaky_Tutors.pdf) or visit the [DOI](https://doi.org/10.1145/3805696.3846495).

## Quick Start

From the project root:

```bash
./bin/install
./bin/run
```

Python 3.10 or newer is required. GPT generation requires an OpenAI API key, and gated Hugging Face models require the appropriate access token. Store credentials in a local `.env` file; never commit them.

## What These Commands Do

- `./bin/install`
  - creates `.venv`
  - installs dependencies
  - uses the bundled `Data/math.json` dataset, rebuilding it only if missing
- `./bin/run`
  - runs `scripts/run_experiment.py`
  - runs `scripts/evaluate_results.py`
  - uses `config/experiment.json` by default

## Output Files

Main outputs are written to the `results_dir` in `config/experiment.json` (default: `results/`):

- `results/<model>/<system>.jsonl`
- `results/<model>/<system>_evaluated.jsonl`
- `results/summary.csv`
- `results/summary.json`

Generated results are intentionally not tracked by Git.

## Config

The main configuration file is `config/experiment.json`. Common fields include:

- `sample_size`
- `models`
- `systems`
- `results_dir`
- `batch_size`

To use another configuration file:

```bash
./bin/run /path/to/your_config.json
```

Use a new, empty `results_dir` when changing experiment settings.

## Dataset

The repository includes the fixed 500-problem MATH subset used by the pipeline, not the full third-party benchmark. It is paired with ten hand-written student prompts, producing 5,000 cases per condition.

To rebuild the subset from the configured Hugging Face mirror:

```bash
.venv/bin/python scripts/prepare_math_dataset.py --output Data/math.json
```

Dataset details and checksums are documented in [Data/README.md](Data/README.md).

## Student Prompts

The prompts in `Data/dataset_b.json` represent realistic student pressure tactics, including direct answer requests, exam-time urgency, yes/no confirmation, and instruction-override attempts. They test whether the tutor avoids final-answer leakage under adversarial as well as cooperative interactions.

## Reproducibility

The fixed experiment inputs are included in `Data/`. Fresh model outputs may vary with provider updates, model revisions, hardware, and package versions. Local results and historical research artifacts are excluded from Git.

## Paper

**Strong Solvers, Leaky Tutors: Evaluating Answer Leakage in LLM Math Tutors**

Aubin Mugisha and Behrooz Mansouri
University of Southern Maine

JCDL '26, October 13–16, 2026, Frisco, Texas, USA
