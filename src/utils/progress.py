"""Progress bar wrapper — uses tqdm if available, else plain range."""

from __future__ import annotations

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def tqdm(iterable, desc: str = "", **kwargs):
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, **kwargs)
    print(desc)
    return iterable
