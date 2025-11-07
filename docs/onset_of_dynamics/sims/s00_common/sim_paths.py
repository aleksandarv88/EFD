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
    def __init__(
        self,
        sim_root: Path | str,
        images_subdir: str = "images",
        logs_subdir: str = "logs",
        results_name: str = "RESULTS.txt",
    ) -> None:
        self.sim_root = Path(sim_root).resolve()
        self.images_dir = self.sim_root / images_subdir
        self.logs_dir = self.sim_root / logs_subdir
        self.results_path = self.sim_root / results_name
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def images_next_version(self) -> Path:
        existing = [
            d for d in self.images_dir.iterdir() if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ]
        n = max([int(d.name[1:]) for d in existing], default=0) + 1
        out = self.images_dir / f"v{n:03d}"
        out.mkdir(parents=True, exist_ok=True)
        return out
