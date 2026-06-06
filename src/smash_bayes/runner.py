from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import subprocess
import brass as br


@dataclass
class SmashRunner:
    smash_exe_path: str
    input_config: str
    output_folder: str
    file_name: str
    analysis_name: str
    output_quantities: list[str]

    parameters: dict[str, Any] = field(default_factory=dict)
    meta_parameters: dict[str, Any] = field(default_factory=dict)

    def output_file(self) -> str:
        return str(Path(self.output_folder) / self.file_name)

    def to_yaml_override(self, key: str, value: Any) -> str:
        parts = key.split(".")

        yaml = f"{parts[-1]}: {value}"

        for part in reversed(parts[:-1]):
            yaml = f"{part}: {{ {yaml} }}"

        return yaml

    def smash_config_args(self) -> list[str]:
        args = [
            "-i",
            self.input_config,
            "-o",
            self.output_folder,
        ]

        for key, value in self.parameters.items():
            args.extend(["-c", self.to_yaml_override(key, value)])

        return args

    def run(self) -> None:
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        command = [self.smash_exe_path] + self.smash_config_args()

        log_file = Path(self.output_folder) / "smash.log"

        with open(log_file, "w") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

    def analyze(self) -> dict[str, Any]:

        result = br.run_analysis_one_file(
            self.output_file(),
            meta="None",
            analysis_name=self.analysis_name,
            quantities=self.output_quantities,
        )

        analysis = br.create_analysis(self.analysis_name)
        analysis.finalize(result)
        return result["None"][self.analysis_name]
