#!/usr/bin/env python3

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import qmc

import smash_bayes
from smash_bayes.studies import BaryonStoppingStudy


PARAMETER_RANGES = {
    "Collision_Term.String_Parameters.Damp_Popcorn": (0.0, 1.0),
    "Collision_Term.String_Parameters.StringZ_A": (0.0, 2.0),
    "Collision_Term.String_Parameters.StringZ_B": (0.2, 2.0),
    "Collision_Term.String_Parameters.StringZ_A_Leading": (0.0, 2.0),
    "Collision_Term.String_Parameters.StringZ_B_Leading": (0.5, 5.0),
    "Collision_Term.String_Parameters.Popcorn_Rate": (0.0, 2.0),
    "Collision_Term.String_Parameters.Strange_Supp": (0.0, 1.0),
    "Collision_Term.String_Parameters.Diquark_Supp": (0.0, 1.0),
    "Collision_Term.String_Parameters.Prob_SQ_to_QQ": (0.0, 1.0),
    "Collision_Term.String_Parameters.Popcorn_Spair": (0.0, 1.0),
    "Collision_Term.String_Parameters.Popcorn_Smeson": (0.0, 1.0),
}


def sample_parameters(parameter_ranges, n_points, seed=None):
    names = list(parameter_ranges)
    bounds = np.array([parameter_ranges[name] for name in names])

    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit_samples = sampler.random(n_points)
    samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

    return [dict(zip(names, row)) for row in samples]


def make_study(args):
    factory = smash_bayes.SmashRunnerFactory(
        smash_exe_path=args.smash_exe,
        input_config=args.input_config,
        output_base=args.output_base,
        file_name=args.file_name,
        analysis_name=args.analysis_name,
        output_quantities=args.output_quantities,
        n_events=args.n_events,
    )

    return BaryonStoppingStudy(factory)


def run_one(job):
    run_id, sqrtsnn, parameters, args = job

    study = make_study(args)

    return study.run_point(
        run_id=run_id,
        sqrtsnn=sqrtsnn,
        extra_parameters=parameters,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Latin-hypercube design for the baryon stopping study."
    )

    parser.add_argument("--smash-exe", default="smash")
    parser.add_argument("--input-config", default="config.yaml")
    parser.add_argument("--output-base", default="runs")

    parser.add_argument("--file-name", default="particles_custom.bin")
    parser.add_argument("--analysis-name", default="baryon_stopping")

    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--n-points", type=int, default=450)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=[17.3],
        help="List of sqrt(s_NN) values.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="Base random seed.",
    )

    parser.add_argument(
        "--output-quantities",
        nargs="+",
        default=["mass", "p0", "pz", "px", "py", "pdg", "ncoll"],
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    jobs = []
    run_id = 0

    for sqrtsnn in args.energies:
        design = sample_parameters(
            PARAMETER_RANGES,
            n_points=args.n_points,
            seed=args.seed + int(100 * sqrtsnn),
        )

        for parameters in design:
            jobs.append((run_id, sqrtsnn, parameters, args))
            run_id += 1

    print(f"Prepared {len(jobs)} jobs")
    print(f"Energies: {args.energies}")
    print(f"Workers: {args.workers}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]
        completed = as_completed(futures)

        if args.progress:
            from tqdm import tqdm

            completed = tqdm(
                completed,
                total=len(futures),
                desc="Running SMASH points",
            )
            write = tqdm.write
        else:
            write = print

        for future in completed:
            try:
                path = future.result()
                write(f"Saved {path}")
            except Exception as error:
                write(f"Run failed: {error}")


if __name__ == "__main__":
    main()
