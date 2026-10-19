"""Command-line interface for the registered Nestor experiments.

Usage:
    nestor_cli --experiment <experiment_name>
    nestor_cli --list-experiments
    torchrun --nproc_per_node=8 -m nestor.cli --experiment nestor_200m
"""

from __future__ import annotations

# Imported first so timing.PROCESS_START reflects process start, letting the
# trainer report MFU "including init" (torch/cuda import, data load, resume,
# torch.compile). Keep this above the heavier imports below.
from nestor.utils import timing as _timing  # noqa: F401

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

from nestor.utils.distributed import cleanup_distributed


def _load_dotenv() -> None:
    """Load environment variables from .env file if present."""
    try:
        from dotenv import load_dotenv

        # Look for .env in current directory and parent directories
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return

        # Try to find .env relative to the package
        package_root = Path(__file__).parent.parent
        env_path = package_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        # python-dotenv not installed, skip
        pass


def _discover_experiments() -> None:
    """
    Import all experiment modules to trigger @register_experiment decorators.

    This discovers and imports all modules under nestor.experiments.
    """
    import nestor.experiments

    assert nestor.experiments.__file__ is not None
    package_path = Path(nestor.experiments.__file__).parent

    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name.startswith("_"):
            continue
        if module_info.name == "registry":
            continue
        try:
            importlib.import_module(f"nestor.experiments.{module_info.name}")
        except ImportError as e:
            print(
                f"Warning: Failed to import experiment module '{module_info.name}': {e}"
            )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Nestor experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run an experiment
    nestor_cli --experiment nestor_200m

    # List available experiments
    nestor_cli --list-experiments
""",
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Name of the experiment to run (function name from @register_experiment)",
    )
    parser.add_argument(
        "--list-experiments",
        "-l",
        action="store_true",
        help="List all available experiments and exit",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI."""
    # Load environment variables first
    _load_dotenv()

    args = _parse_args()

    # Import after dotenv is loaded
    from nestor.configs.jobs import EvalJobConfig, TrainJobConfig
    from nestor.experiments.registry import get_experiment, list_experiments
    from nestor.pipelines.eval import EvalPipeline
    from nestor.pipelines.train import TrainPipeline

    # Discover and register experiments
    _discover_experiments()

    # Handle --list-experiments
    if args.list_experiments:
        experiments = list_experiments()
        if experiments:
            print("Available experiments:")
            for name in experiments:
                print(f"  - {name}")
        else:
            print("No experiments registered.")
            print(
                "Define experiments in nestor/experiments/ using @register_experiment"
            )
        return 0

    # Require --experiment if not listing
    if not args.experiment:
        print(
            "Error: --experiment is required. Use --list-experiments to see available options."
        )
        return 1

    # Get the experiment function
    try:
        experiment_fn = get_experiment(args.experiment)
    except KeyError as e:
        print(f"Error: {e}")
        return 1

    # Build the experiment config
    print(f"Running experiment: {args.experiment}")
    experiment_config = experiment_fn()

    # Override experiment name with function name
    experiment_config.name = args.experiment

    # Run each job in the experiment
    try:
        for i, job_config in enumerate(experiment_config.jobs):
            print(f"\n{'=' * 60}")
            print(
                f"Job {i + 1}/{len(experiment_config.jobs)}: {job_config.name}"
            )
            print(f"{'=' * 60}\n")

            # Dispatch to appropriate pipeline
            if isinstance(job_config, TrainJobConfig):
                pipeline = TrainPipeline.from_config(
                    job_config=job_config,
                    experiment_config=experiment_config,
                )
                pipeline.run()
            elif isinstance(job_config, EvalJobConfig):
                pipeline = EvalPipeline.from_config(
                    job_config=job_config,
                    experiment_config=experiment_config,
                )
                pipeline.run()
            else:
                print(
                    f"Warning: Unknown job type '{job_config.job_type}', skipping."
                )

        print(f"\n{'=' * 60}")
        print("Experiment complete!")
        print(f"{'=' * 60}")

    finally:
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    sys.exit(main())
