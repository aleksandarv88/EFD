# docs/onset_of_dynamics/sims/s00_common/parallel.py
from __future__ import annotations
import os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- robust import: works when run as package OR as plain script ---
try:
    # package-style
    from .progress import pbar
except Exception:
    # script-style fallback
    HERE = os.path.dirname(__file__)
    if HERE and HERE not in sys.path:
        sys.path.append(HERE)
    from progress import pbar  # type: ignore

# tiny in-proc object store for workers
_GLOBAL_STORE = {}

def _init_worker(
    seed_offset: int = 0,
    np_seed: int | None = None,
    graph_path: str | None = None,
    graph_key: str | None = None,
    matrix_key: str | None = None,
    matrix_path: str | None = None,
):
    """
    Initializer for ProcessPool workers.
    - Optionally seeds NumPy RNG (np_seed + seed_offset).
    - Optionally loads a pickled NetworkX graph into _GLOBAL_STORE[graph_key].
    - Optionally loads a SciPy sparse CSR matrix into _GLOBAL_STORE[matrix_key].
    """
    if np_seed is not None:
        try:
            import numpy as _np
            _np.random.seed(np_seed + seed_offset)
        except Exception:
            pass

    if graph_path and graph_key:
        try:
            import pickle as _pkl
            with open(graph_path, "rb") as f:
                G = _pkl.load(f)
            _GLOBAL_STORE[graph_key] = G
        except Exception:
            _GLOBAL_STORE[graph_key] = None

    if matrix_path and matrix_key:
        try:
            from scipy.sparse import load_npz as _load_npz
            A = _load_npz(matrix_path)
            _GLOBAL_STORE[matrix_key] = A
        except Exception:
            _GLOBAL_STORE[matrix_key] = None

def get_global(key: str):
    return _GLOBAL_STORE.get(key, None)

def map_process(
    fn,
    tasks,
    max_workers: int | None = None,
    desc: str | None = None,
    initializer=None,
    initargs: tuple = (),
):
    """
    Submit tasks (iterable) to a ProcessPool and preserve input order.
    Returns a list with the same length as 'tasks', where each element
    is either the result or an Exception (caller can raise if desired).
    """
    tasks = list(tasks)
    n = len(tasks)
    results = [None] * n
    if n == 0:
        return results

    with ProcessPoolExecutor(max_workers=max_workers, initializer=initializer, initargs=initargs) as ex:
        fut_to_idx = {ex.submit(fn, tasks[i]): i for i in range(n)}
        with pbar(total=n, desc=(desc or "process")) as bar:
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = e
                bar.update(1)
    return results
