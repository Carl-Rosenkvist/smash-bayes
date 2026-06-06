from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class EmulatorDataset:
    X: pd.DataFrame
    Y: pd.DataFrame

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.X.to_parquet(directory / "X.parquet")
        self.Y.to_parquet(directory / "Y.parquet")

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)

        return cls(
            X=pd.read_parquet(directory / "X.parquet"),
            Y=pd.read_parquet(directory / "Y.parquet"),
        )
