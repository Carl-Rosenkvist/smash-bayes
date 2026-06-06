from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smash_bayes.runner import SmashRunner


@dataclass
class SmashRunnerFactory:
    smash_exe_path: str
    input_config: str

    output_base: str

    file_name: str
    analysis_name: str
    output_quantities: list[str]

    n_events: int

    def make_runner(
        self,
        parameters: dict[str, Any],
        run_id: int | None = None,
        trial_name: str | None = None,
        n_events: int | None = None,
        meta_parameters: dict[str, Any] | None = None,
    ) -> SmashRunner:
        """
        Create a SmashRunner for a single parameter point.

        Either run_id or trial_name may be supplied.
        """

        if run_id is not None:
            trial_name = f"run_{run_id:05d}"

        if trial_name is None:
            raise ValueError("Either run_id or trial_name must be specified.")

        output_folder = Path(self.output_base) / trial_name

        runner_parameters = {
            "General.Nevents": (self.n_events if n_events is None else n_events),
            "Output.Particles.Quantities": self.output_quantities,
            **parameters,
        }

        return SmashRunner(
            smash_exe_path=self.smash_exe_path,
            input_config=self.input_config,
            output_folder=str(output_folder),
            file_name=self.file_name,
            analysis_name=self.analysis_name,
            output_quantities=self.output_quantities,
            parameters=runner_parameters,
            meta_parameters=meta_parameters or parameters,
        )
