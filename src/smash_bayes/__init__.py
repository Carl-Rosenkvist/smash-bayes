# src/smash_bayes/__init__.py

from .runner import SmashRunner
from .runner_factory import SmashRunnerFactory
from .emulator import GaussianPCAEmulator

__version__ = "0.1.0"

__all__ = [
    "SmashRunner",
    "SmashRunnerFactory",
    "GaussianPCAEmulator",
]
