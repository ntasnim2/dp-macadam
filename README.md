# DP-MacAdam Python Implementation

This repository contains a modular version of the original `dp-macadam.py` script for DP-MaCAdam MNIST experiments. The original script is kept at the repo root, and the package-based implementation lives in `dp_macadam/`.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python scripts/run_experiment.py
```

The default run matches the quick single-script setup:

- seed: `42`
- sigma: `0.5`
- epochs: `5`
- batch size: `256`

You can override experiment settings from the command line:

```bash
python scripts/run_experiment.py --epochs 20 --sigmas 0.5 1.0 2.0 --seeds 42 123 456
```
