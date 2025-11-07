from __future__ import annotations
import os, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Sequence

try:
    from progress import pbar
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def pbar(total=None, desc=""):
        class _N:
            def update(self, n=1): pass
        yield _N()

_GLOB: dict[str, Any] = {}

def _init_worker(seed_offset: int = 0, graph_bytes: bytes | None = None,
                 graph_path: str | None = None, key: str = "_G") -> None:
    if graph_bytes is not None:
        _GLOB[key] = pickle.loads(graph_bytes)
    elif graph_path is not None:
        with open(graph_path, "rb") as f:
            _GLOB[key] = pickle.load(f)
    try:
        import random
        random.seed((os.getpid() ^ seed_offset) & 0xFFFFFFFF)
    except Exception:
        pass

def get_global(key: str) -> Any:
    return _GLOB[key]

def map_process(
    fn: Callable[[Any], Any],
    tasks: Sequence[Any],
    *,
    max_workers: int | None = None,
    desc: str = "work",
    initializer=None,
    initargs: tuple[Any, ...] = (),
) -> list[Any]:
    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = max(1, cpu - 1)

    results: list[Any] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=initializer, initargs=initargs) as ex:
        fut_to_idx = {ex.submit(fn, tasks[i]): i for i in range(len(tasks))}
        with pbar(total=len(fut_to_idx), desc=desc) as bar:
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = e
                bar.update(1)
    return results
