# Onset_Of_Dynamics/sims/00_common/sim_paths.py
from __future__ import annotations
from pathlib import Path

def sim_root(file_dunder: str) -> Path:
    return Path(file_dunder).resolve().parent

def results_path(sim_root_dir: Path, sim_prefix: str) -> Path:
    return sim_root_dir / f"{sim_prefix}_RESULTS.txt"

def images_root(sim_root_dir: Path) -> Path:
    return sim_root_dir / "images"

def logs_root(sim_root_dir: Path) -> Path:
    d = sim_root_dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
