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

class SimPaths:
    def __init__(self, sim_root, images_subdir="images", logs_subdir="logs", results_name="RESULTS.txt"):
        self.sim_root = os.path.abspath(sim_root)
        self.images_dir = os.path.join(self.sim_root, images_subdir)
        self.logs_dir = os.path.join(self.sim_root, logs_subdir)
        self.results_path = os.path.join(self.sim_root, results_name)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def images_next_version(self) -> str:
        existing = [d for d in os.listdir(self.images_dir) if d.startswith("v") and d[1:].isdigit()]
        n = max([int(d[1:]) for d in existing], default=0) + 1
        out = os.path.join(self.images_dir, f"v{n:03d}")
        os.makedirs(out, exist_ok=True)
        return out
