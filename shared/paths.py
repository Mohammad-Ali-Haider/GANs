from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    exp_dir: Path
    models_dir: Path
    samples_dir: Path
    datasets_dir: Path

def find_project_root(start: Path | None = None) -> Path:
    """
    Find GAN_Learning root by walking upwards until pyproject.toml is found.
    Works even when notebooks run inside experiments/*.
    """
    start = start or Path.cwd()
    cur = start.resolve()

    for _ in range(20):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    raise RuntimeError("Could not find project root (pyproject.toml). Run from inside GAN_Learning/ or its subfolders.")

def get_experiment_paths(experiment_name: str) -> ExperimentPaths:
    """
    experiment_name like:
      "01_mlp_gan" or "02_dcgan" or "03_cdcgan"
    """
    root = find_project_root()
    exp_dir = root / "experiments" / experiment_name
    models_dir = exp_dir / "models"
    samples_dir = exp_dir / "samples"
    datasets_dir = root / "datasets"

    models_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        root=root,
        exp_dir=exp_dir,
        models_dir=models_dir,
        samples_dir=samples_dir,
        datasets_dir=datasets_dir,
    )