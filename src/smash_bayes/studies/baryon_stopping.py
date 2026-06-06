from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smash_bayes.runner_factory import SmashRunnerFactory


M_N = 0.9382720813

PDG_GROUPS = {
    3122: [3122, 3212],
    -3122: [-3122, -3212],
}

DEFAULT_PDGS = [-211, 2212, -2212, 211, 3122, 321, -321, 3312]


def members(pdg: int) -> list[int]:
    return PDG_GROUPS.get(pdg, [pdg])


def spectrum(
    block: dict[str, Any],
    pdg: int,
    sqrtsnn: float,
) -> dict[str, np.ndarray] | None:
    groups = [
        block["per_pdg"][member]
        for member in members(pdg)
        if member in block["per_pdg"]
    ]

    if not groups:
        return None

    y = groups[0]["y"]
    pz = groups[0]["pz"]

    dndy = sum(group["dndy"] for group in groups)
    dndpz = sum(group["dndpz"] for group in groups)

    pbeam = np.sqrt(max(sqrtsnn**2 / 4.0 - M_N**2, 0.0))

    xF = pz / pbeam
    dndxF = dndpz * pbeam

    weights = np.array([group["dndpz"] for group in groups])
    mean_pts = np.array([group["mean_pt_vs_pz"] for group in groups])
    weight_sum = weights.sum(axis=0)

    mean_pt = np.divide(
        (weights * mean_pts).sum(axis=0),
        weight_sum,
        out=np.zeros_like(weight_sum),
        where=weight_sum > 0,
    )

    return {
        "y": y,
        "dndy": dndy,
        "xF": xF,
        "dndxF": dndxF,
        "mean_pt": mean_pt,
    }


def flatten_spectrum(
    spec: dict[str, np.ndarray],
    pdg: int,
) -> dict[str, float]:
    flat = {}

    for quantity in ["dndy", "dndxF", "mean_pt"]:
        values = spec[quantity]

        for i, value in enumerate(values):
            flat[f"{pdg}_{quantity}_bin_{i}"] = float(value)

    return flat


def get_observable(
    row: pd.Series,
    pdg: int,
    quantity: str,
) -> np.ndarray:
    prefix = f"{pdg}_{quantity}_bin_"

    columns = sorted(
        [column for column in row.index if column.startswith(prefix)],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )

    if not columns:
        raise KeyError(f"No columns found for {prefix}")

    return row[columns].to_numpy(dtype=float)


def get_dndy(row: pd.Series, pdg: int) -> np.ndarray:
    return get_observable(row, pdg, "dndy")


def get_dndxF(row: pd.Series, pdg: int) -> np.ndarray:
    return get_observable(row, pdg, "dndxF")


def get_mean_pt(row: pd.Series, pdg: int) -> np.ndarray:
    return get_observable(row, pdg, "mean_pt")


@dataclass
class BaryonStoppingStudy:
    factory: SmashRunnerFactory
    pdgs: list[int] = None

    def __post_init__(self) -> None:
        if self.pdgs is None:
            self.pdgs = DEFAULT_PDGS

    def make_row(
        self,
        result: dict[str, Any],
        parameters: dict[str, Any],
        sqrtsnn: float,
    ) -> dict[str, float]:
        row = dict(parameters)

        for pdg in self.pdgs:
            spec = spectrum(result, pdg, sqrtsnn)

            if spec is not None:
                row.update(flatten_spectrum(spec, pdg))

        return row

    def run_point(
        self,
        run_id: int,
        sqrtsnn: float,
        extra_parameters: dict[str, Any] | None = None,
    ) -> Path:
        parameters = {
            "Modi.Collider.Sqrtsnn": sqrtsnn,
        }

        if extra_parameters is not None:
            parameters.update(extra_parameters)

        runner = self.factory.make_runner(
            parameters=parameters,
            run_id=run_id,
        )

        runner.run()
        result = runner.analyze()

        row = self.make_row(
            result=result,
            parameters=parameters,
            sqrtsnn=sqrtsnn,
        )

        parquet_dir = Path(self.factory.output_base) / "parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)

        path = parquet_dir / f"run_{run_id:05d}.parquet"

        pd.DataFrame([row]).to_parquet(path, index=False)

        return path
