# Onset_Of_Dynamics/sims/00_common/sim_results.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable
from lib.efd.io_utils import run_header, env_block

def parse_doc_meta(doc_path: Path) -> dict:
    """
    Read first ~30 lines of a Stage doc and extract Version and Doc SHA256 if present.
    Expects the ASCII META header we defined earlier.
    """
    meta = {"Version": "n/a", "DocSHA": "n/a", "Title": doc_path.name}
    if not doc_path.exists():
        return meta
    try:
        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 30: break
                if line.lower().startswith("version:"):
                    meta["Version"] = line.split(":", 1)[1].strip()
                if line.lower().startswith("doc sha256:"):
                    meta["DocSHA"] = line.split(":", 1)[1].strip().split()[0]
                if line.lower().startswith("title:"):
                    meta["Title"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return meta

def write_run_block(
    results_txt: Path,
    header_line: str,
    stages_meta: dict[str, dict],  # {"S1": {"Version":..., "DocSHA":...}, ...}
    config_dump: str,
    metrics_lines: Iterable[str],
    image_paths: Iterable[Path],
):
    results_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(results_txt, "a", encoding="utf-8") as f:
        f.write(header_line + "\n")
        # stages line
        versions = " | ".join([f"{k} {v.get('Version','n/a')}" for k,v in stages_meta.items()])
        f.write(f"STAGES: {versions}\n")
        # shas line
        shas = " ".join([f"{k}={v.get('DocSHA','n/a')}" for k,v in stages_meta.items()])
        f.write(f"DOC SHAS: {shas}\n")
        # config
        f.write("CONFIG:\n")
        f.write(indent_block(config_dump, prefix="  "))
        # metrics
        f.write("METRICS:\n")
        for m in metrics_lines:
            f.write(f"  - {m}\n")
        # images
        f.write("IMAGES:\n")
        for p in image_paths:
            f.write(f"  {p.as_posix()}\n")
        # env
        f.write(env_block() + "\n")
        f.write("=== END RUN ===\n\n")

def indent_block(s: str, prefix: str = "  ") -> str:
    return "".join(prefix + line if line.strip() else line for line in s.splitlines(True))
