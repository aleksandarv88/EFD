# lib/efd/io_utils.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import hashlib
import platform
import sys
import re

_VPAT = re.compile(r"^v(\d{3})$")

def ensure_version_dir(images_root: Path) -> Path:
    images_root.mkdir(parents=True, exist_ok=True)
    existing = [d for d in images_root.iterdir() if d.is_dir() and _VPAT.match(d.name)]
    v = f"v{(max([int(_VPAT.match(d.name).group(1)) for d in existing]) + 1) if existing else 1:03d}"
    out = images_root / v
    out.mkdir(parents=True, exist_ok=True)
    return out  # e.g. images/v003

def next_fig_path(version_dir: Path, prefix: str = "fig", idx: int | None = None, ext: str = ".png") -> Path:
    if idx is None:
        count = sum(1 for _ in version_dir.glob(f"{prefix}_*{ext}"))
        idx = count + 1
    return version_dir / f"{prefix}_{idx:03d}{ext}"

def sha256_of_file(path: Path, first_n: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if first_n:
            h.update(f.read(first_n))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()

def run_header(version_dir: Path, seed: int | None = None, commit: str | None = None, code_sha: str | None = None) -> str:
    iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"=== RUN {iso} | {version_dir.as_posix()} | seed={seed} | commit={commit} | code_sha={code_sha} ==="

def env_block() -> str:
    try:
        import numpy, networkx, matplotlib  # type: ignore
        npv, nxv, mplv = numpy.__version__, networkx.__version__, matplotlib.__version__
    except Exception:
        npv = nxv = mplv = "n/a"
    pyv = sys.version.split()[0]
    osdesc = f"{platform.system()}-{platform.release()}-{platform.machine()}"
    return f"ENV:\n  python={pyv} | numpy={npv} | networkx={nxv} | matplotlib={mplv}\n  os={osdesc}"
