# Tutor

This project runs the full tutoring experiment and then evaluates the results.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Fill in `.env`:

```env
OPENAI_API_KEY=...
HF_TOKEN=...
```

## Run The Experiment

```bash
python run_experiment.py
```

This runs all models and all systems.

It uses:

- `Data/math.json`
- `Data/dataset_b.json`

It writes results to `results/`.

## Evaluate Everything

```bash
python evaluate_results.py
```

This evaluates everything in `results/`.

It writes:

- `results/<model>/<system>_evaluated.jsonl`
- `results/summary.csv`
- `results/summary.json`
