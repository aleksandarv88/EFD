# lib/efd/io_utils.py
from pathlib import Path
import re
from datetime import datetime

_vpat = re.compile(r"^v(\d{3})$")

def ensure_version_dir(images_root: Path) -> Path:
    images_root.mkdir(parents=True, exist_ok=True)
    existing = [d for d in images_root.iterdir() if d.is_dir() and _vpat.match(d.name)]
    if not existing:
        v = "v001"
    else:
        mx = max(int(_vpat.match(d.name).group(1)) for d in existing)
        v = f"v{mx+1:03d}"
    out = images_root / v
    out.mkdir(parents=True, exist_ok=True)
    return out  # e.g. images/v003/

def next_fig_path(version_dir: Path, prefix: str = "fig", idx: int | None = None, ext: str = ".png") -> Path:
    if idx is None:
        # auto-increment by counting existing files with this prefix
        n = sum(1 for _ in version_dir.glob(f"{prefix}_*{ext}"))
        idx = n + 1
    return version_dir / f"{prefix}_{idx:03d}{ext}"

def run_header(version_dir: Path, seed: int | None = None, commit: str | None = None, code_sha: str | None = None) -> str:
    iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"=== RUN {iso} | {version_dir.as_posix()} | seed={seed} | commit={commit} | code_sha={code_sha} ==="
