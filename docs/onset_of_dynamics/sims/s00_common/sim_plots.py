# Onset_Of_Dynamics/sims/00_common/sim_plots.py
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from lib.efd.viz import setup_matplotlib, save_png, save_svg, save_mp4
from lib.efd.io_utils import ensure_version_dir, next_fig_path

def start_fig_env(images_root: Path) -> Path:
    setup_matplotlib()                # dpi=180, tight bbox, fonts
    vdir = ensure_version_dir(images_root)  # images/vNNN
    return vdir

def save_fig(fig: plt.Figure, vdir: Path, prefix: str="fig") -> Path:
    p = next_fig_path(vdir, prefix=prefix, ext=".png")
    save_png(fig, p)
    return p

def save_fig_svg(fig: plt.Figure, vdir: Path, prefix: str="fig") -> Path:
    p = next_fig_path(vdir, prefix=prefix, ext=".svg")
    save_svg(fig, p)
    return p

def save_anim_mp4(frames, vdir: Path, prefix: str="anim", fps: int = 12, crf: int = 27) -> Path:
    p = next_fig_path(vdir, prefix=prefix, ext=".mp4")
    save_mp4(frames, p, fps=fps, crf=crf)
    return p
