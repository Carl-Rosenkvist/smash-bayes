#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import smash_bayes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Gaussian PCA emulator from parquet run outputs."
    )

    parser.add_argument(
        "--runs-dir", type=Path)
    )
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--emulator-path", type=Path, default=None)

    parser.add_argument(
        "--variance-fraction",
        type=float,
        default=0.99,
        help="Fraction of observable variance to keep in PCA, e.g. 0.99.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.variance_fraction <= 1.0):
        raise ValueError("--variance-fraction must be in the interval (0, 1].")

    runs_dir = args.runs_dir
    parquet_dir = args.parquet_dir or runs_dir / "parquet"
    dataset_dir = args.dataset_dir or runs_dir / "dataset"
    emulator_path = args.emulator_path or runs_dir / "emulator" / "gp_emulator.pkl"

    files = sorted(parquet_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    df = pd.concat(
        [pd.read_parquet(file) for file in files],
        ignore_index=True,
    )

    parameter_columns = [column for column in df.columns if "_bin_" not in column]
    observable_columns = [column for column in df.columns if "_bin_" in column]

    X = df[parameter_columns]
    Y = df[observable_columns]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    emulator_path.parent.mkdir(parents=True, exist_ok=True)

    X.to_parquet(dataset_dir / "X.parquet", index=False)
    Y.to_parquet(dataset_dir / "Y.parquet", index=False)

    print(f"Loaded {len(files)} parquet files")
    print(f"Total rows: {len(df)}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    if len(X) < 3:
        raise ValueError("Need at least 3 runs to train the GP emulator.")

    max_components = min(len(X) - 1, Y.shape[1])

    print(f"Maximum valid PCA components: {max_components}")
    print(f"Target retained variance fraction: {args.variance_fraction}")

    emulator = smash_bayes.GaussianPCAEmulator.train(
        X,
        Y,
        n_components=args.variance_fraction,
    )

    emulator.save(emulator_path)

    print(f"Saved dataset to {dataset_dir}")
    print(f"Saved emulator to {emulator_path}")


if __name__ == "__main__":
    main()
