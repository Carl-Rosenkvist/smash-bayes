# smash-bayes

A lightweight framework for Bayesian calibration and emulator construction for SMASH transport model studies.

## Installation

Clone the repository:

bash git clone https://github.com/Carl-Rosenkvist/smash-bayes.git cd smash-bayes 

Install in editable mode:

bash python3 -m pip install -e . 

## Running a study

The package currently provides a baryon stopping study based on Latin-hypercube sampling of SMASH string fragmentation parameters.

Run a design:

bash smash-bayes-run-baryon-stopping \     --smash-exe smash \     --input-config config.yaml \     --output-base runs \     --n-points 500 \     --n-events 10000 \     --workers 32 

This creates parquet files containing the observables extracted from each SMASH run.

## Training an emulator

Train a Gaussian-process emulator from the generated parquet files:

bash smash-bayes-train-emulator \     --runs-dir runs \     --variance-fraction 0.99 

Outputs:

text runs/ ├── parquet/ ├── dataset/ │   ├── X.parquet │   └── Y.parquet └── emulator/     └── gp_emulator.pkl 

The emulator uses:

- Standard scaling of parameters and observables
- PCA compression of observables
- Independent Gaussian processes for PCA coefficients

The number of PCA components is automatically chosen to retain the requested fraction of the observable variance.

## Package structure

text src/smash_bayes/ ├── studies/ ├── emulator.py ├── runner.py └── cli/ 

Additional studies can be implemented by deriving from the existing study infrastructure and exposing new CLI entry points through pyproject.toml.
