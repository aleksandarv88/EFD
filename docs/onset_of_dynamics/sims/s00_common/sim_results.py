# Onset_Of_Dynamics/sims/00_common/sim_results.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

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
                if i > 30:
                    break
                if line.lower().startswith("version:"):
                    meta["Version"] = line.split(":", 1)[1].strip()
                if line.lower().startswith("doc sha256:"):
                    meta["DocSHA"] = line.split(":", 1)[1].strip().split()[0]
                if line.lower().startswith("title:"):
                    meta["Title"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return meta


class ResultsWriter:
    """Append structured blocks to *_RESULTS.txt with consistent headers."""

    def __init__(self, results_txt: Path, stage_docs: Mapping[str, Path]):
        self.results_txt = Path(results_txt)
        self.stage_docs = {k: Path(v) for k, v in stage_docs.items()}

    def append_run_block(
        self,
        version_dir: Path,
        config_snapshot: dict,
        metrics: dict,
        images: Iterable[Path],
        *,
        seed: int | None = None,
        commit: str | None = None,
        code_sha: str | None = None,
    ) -> None:
        header_line = run_header(version_dir, seed=seed, commit=commit, code_sha=code_sha)
        stages_meta = {k: parse_doc_meta(path) for k, path in self.stage_docs.items()}
        config_dump = json.dumps(config_snapshot, indent=2, default=_json_default)
        metric_lines = [f"{k}={v}" for k, v in metrics.items() if k != "images"]
        write_run_block(
            self.results_txt,
            header_line,
            stages_meta,
            config_dump,
            metric_lines,
            images,
        )


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
        versions = " | ".join([f"{k} {v.get('Version','n/a')}" for k, v in stages_meta.items()])
        f.write(f"STAGES: {versions}\n")
        # shas line
        shas = " ".join([f"{k}={v.get('DocSHA','n/a')}" for k, v in stages_meta.items()])
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


def _json_default(obj):
    if isinstance(obj, Path):
        return obj.as_posix()
    return str(obj)
