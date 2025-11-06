# lib/efd/viz.py
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio  # only if you export mp4/gif

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
    fig.savefig(path, dpi=mpl.rcParams["savefig.dpi"], bbox_inches="tight", pad_inches=0.05)

def save_svg(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.05)

def _drop_alpha(frame: np.ndarray) -> np.ndarray:
    # frame shape (H,W,3 or 4); convert RGBA→RGB to avoid alpha in mp4
    if frame.ndim == 3 and frame.shape[-1] == 4:
        return frame[..., :3]
    return frame

def save_mp4(frames: list[np.ndarray], path: Path, fps: int = 12, crf: int = 27, preset: str = "medium"):
    """
    frames: list of HxWx3/4 uint8 arrays
    Compact H.264: CRF 26–28 ≈ small files, visually fine.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [_drop_alpha(f) for f in frames]
    writer = imageio.get_writer(
        path.with_suffix(".mp4"),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=[
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",   # broad compatibility
            "-movflags", "+faststart"
        ],
    )
    for f in clean:
        writer.append_data(f)
    writer.close()

def save_gif(frames: list[np.ndarray], path: Path, fps: int = 12):
    # Only if you really need GIFs (bigger than mp4)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path.with_suffix(".gif"), [_drop_alpha(f) for f in frames], duration=1.0/max(fps,1))
