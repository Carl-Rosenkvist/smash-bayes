import smash_bayes
import numpy as np
import pandas as pd
from pathlib import Path

pdg_groups = {
    3122: [3122, 3212],
    -3122: [-3122, -3212],
}

pdgs = [-211, 2212, -2212, 211, 3122, 321, -321, 3312]
M_N = 0.9382720813


def members(pdg):
    return pdg_groups.get(pdg, [pdg])


def spectrum(block, pdg, energy):
    gs = [block["per_pdg"][m] for m in members(pdg) if m in block["per_pdg"]]
    if not gs:
        return None
    y = gs[0]["y"]
    pz = gs[0]["pz"]

    dndy = sum(g["dndy"] for g in gs)
    dndpz = sum(g["dndpz"] for g in gs)

    pbeam = np.sqrt(max(energy**2 / 4 - M_N**2, 0.0))
    xF = pz / pbeam
    dndxF = dndpz * pbeam

    weights = np.array([g["dndpz"] for g in gs])
    pts = np.array([g["mean_pt_vs_pz"] for g in gs])
    wsum = weights.sum(axis=0)

    pt = np.divide(
        (weights * pts).sum(axis=0),
        wsum,
        out=np.zeros_like(wsum),
        where=wsum > 0,
    )

    return {"y": y, "dndy": dndy, "xF": xF, "dndxF": dndxF, "mean_pt": pt}


def flatten_spectrum(
    spectrum: dict[str, np.ndarray],
    pdg: int,
) -> dict[str, float]:
    flat = {}

    for quantity in ["dndy", "dndxF", "mean_pt"]:
        values = spectrum[quantity]

        for i, value in enumerate(values):
            flat[f"{pdg}_{quantity}_bin_{i}"] = float(value)

    return flat


SQRTSNN = 6.3
factory = smash_bayes.SmashRunnerFactory(
    smash_exe_path="/Users/carl/Phd/smash-devel/build/smash",
    input_config="/Users/carl/Phd/smash-bayes/input/pp.yaml",
    output_base="/Users/carl/Phd/smash-bayes/runs",
    file_name="particles_custom.bin",
    analysis_name="baryon_stopping",
    output_quantities=["mass", "p0", "pz", "px", "py", "pdg", "ncoll"],
    n_events=10000,
)
run_id = 2
parms = {"Modi.Collider.Sqrtsnn": SQRTSNN}
smash_runner = factory.make_runner(parameters=parms, run_id=run_id)
smash_runner.run()


result = smash_runner.analyze()
proton_spectrum = spectrum(result, 2212, SQRTSNN)


row = {
    **parms,
    **flatten_spectrum(spectrum(result, 2212, SQRTSNN), 2212),
    **flatten_spectrum(spectrum(result, 211, SQRTSNN), 211),
}

df = pd.DataFrame([row])

parquet_dir = Path(factory.output_base) / "parquet"

parquet_dir.mkdir(parents=True, exist_ok=True)


df.to_parquet(
    parquet_dir / f"run_{run_id:05d}.parquet",
    index=False,
)
