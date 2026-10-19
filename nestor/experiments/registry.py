"""
Experiment registry for config-first training.

Experiments are Python functions decorated with @register_experiment that return
an ExperimentConfig. The function name becomes the experiment name.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from nestor.configs.experiment import ExperimentConfig

# Global registry: function_name -> experiment_function
_EXPERIMENT_REGISTRY: dict[str, Callable[[], "ExperimentConfig"]] = {}


def register_experiment(
    fn: Callable[[], "ExperimentConfig"],
) -> Callable[[], "ExperimentConfig"]:
    """
    Decorator to register an experiment function.

    The function name is used as the experiment name in the registry.

    Args:
        fn (Callable[[], ExperimentConfig]): A function that takes no arguments
            and returns an ExperimentConfig.

    Returns:
        Callable[[], ExperimentConfig]: The same function, unmodified.

    Raises:
        ValueError: If an experiment with the same name is already registered.

    Example:
        @register_experiment
        def nestor_200m() -> ExperimentConfig:
            return ExperimentConfig(jobs=[...])
    """
    assert hasattr(fn, "__name__") and isinstance(fn.__name__, str)
    name = fn.__name__
    if name in _EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment '{name}' is already registered.")
    _EXPERIMENT_REGISTRY[name] = fn
    return fn


def get_experiment(name: str) -> Callable[[], "ExperimentConfig"]:
    """
    Retrieve a registered experiment function by name.

    Args:
        name (str): The name of the experiment (i.e., the function name).

    Returns:
        Callable[[], ExperimentConfig]: The experiment function.

    Raises:
        KeyError: If no experiment with the given name is registered.
    """
    if name not in _EXPERIMENT_REGISTRY:
        available = ", ".join(sorted(_EXPERIMENT_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Experiment '{name}' not found. Available: {available}")
    return _EXPERIMENT_REGISTRY[name]


def list_experiments() -> list[str]:
    """
    List all registered experiment names.

    Returns:
        list[str]: Sorted list of registered experiment names.
    """
    return sorted(_EXPERIMENT_REGISTRY.keys())
