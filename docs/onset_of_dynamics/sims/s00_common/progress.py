from __future__ import annotations
from contextlib import contextmanager
import sys, time

class _Bar:
    def __init__(self, total=None, desc=""):
        self.total = total
        self.desc = desc
        self.n = 0
        self._last = time.time()

    def update(self, n=1):
        self.n += n
        now = time.time()
        if now - self._last >= 0.1 or (self.total and self.n >= self.total):
            if self.total:
                pct = (100.0 * self.n / max(1, self.total))
                sys.stdout.write(f"\r{self.desc} [{self.n}/{self.total}] {pct:5.1f}%")
            else:
                sys.stdout.write(f"\r{self.desc} {self.n}")
            sys.stdout.flush()
            self._last = now
        if self.total and self.n >= self.total:
            sys.stdout.write("\n"); sys.stdout.flush()

@contextmanager
def pbar(total=None, desc=""):
    bar = _Bar(total=total, desc=desc)
    try:
        yield bar
    finally:
        # always finalize cleanly, even on exceptions
        if total is not None and bar.n < total:
            sys.stdout.write(f"\r{desc} [{bar.n}/{total}] ...\n")
            sys.stdout.flush()
