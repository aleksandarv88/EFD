# lib/efd/viz.py
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# imageio import (2.x or 3.x)
try:
    import imageio.v2 as imageio  # type: ignore
except Exception:
    import imageio  # type: ignore

def setup_matplotlib(dpi: int = 180, font: str = "DejaVu Sans"):
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.family": font,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

def save_png(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)

def save_svg(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.05)

def _drop_alpha(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 4:
        return frame[..., :3]
    return frame

def save_mp4(frames: list[np.ndarray], path: Path, fps: int = 12, crf: int = 27, preset: str = "medium"):
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [_drop_alpha(f) for f in frames]
    writer = imageio.get_writer(
        path.with_suffix(".mp4"),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    for f in clean:
        writer.append_data(f)
    writer.close()
