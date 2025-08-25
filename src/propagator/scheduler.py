"""Lightweight event scheduler for propagation updates.

Stores future updates grouped by simulation time and exposes utilities to push
events, pop the earliest batch, and inspect active realizations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Iterable,
    Iterator,
    List,
    Tuple,
    Optional,
    TypeVar,
    Generic
)
from sortedcontainers import SortedDict
import numpy as np
import numpy.typing as npt

# ---- Type aliases -----------------------------------------------------------

TTime = TypeVar("TTime", int, float)  # time keys can be int or float

# Integer coords array of shape (n, 3). We can’t encode the shape statically
# with stdlib typing, but we DO lock the dtype to integer families.
CoordsArray = npt.NDArray[np.integer]

# The payload shape we pass around
UpdateBatch = List[CoordsArray]
ScheduledPair = Tuple[TTime, CoordsArray]
PopResult = Tuple[TTime, UpdateBatch]


def _validate_coords(coords: npt.ArrayLike) -> CoordsArray:
    if not isinstance(coords, np.ndarray):
        raise TypeError("coords must be a numpy.ndarray")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3)")
    if not np.issubdtype(coords.dtype, np.integer):
        raise TypeError("coords dtype must be an integer type")
    # Narrow the runtime type to CoordsArray for the type checker
    return coords  # type: ignore[return-value]


@dataclass
class Scheduler(Generic[TTime]):
    """
    Lightweight event scheduler for propagation updates.

    Generic over the time key type (int or float), so your inputs and outputs
    stay consistent.
    """

    _queue: SortedDict = field(default_factory=SortedDict, init=False, repr=False)

    # --- Basic queue ops -----------------------------------------------------

    def push(self, coords: CoordsArray, time: TTime) -> None:        
        bucket: UpdateBatch = self._queue.setdefault(time, []) # type: ignore
        bucket.append(_validate_coords(coords))

    def push_all(self, updates: Iterable[ScheduledPair[TTime]]) -> None:
        for t, u in updates:
            self.push(u, t)

    def pop(self) -> PopResult[TTime]:
        if not self:
            raise IndexError("pop from empty Scheduler")
        time, updates = self._queue.popitem(index=0)
        return time, list(updates)

    # --- Introspection -------------------------------------------------------

    def active(self) -> npt.NDArray[np.integer]:
        if not self:
            return np.array([], dtype=int)
        arrays = [a for batches in self._queue.values() for a in batches]
        if len(arrays) == 1:
            return np.unique(arrays[0][:, 2])
        stacked = np.concatenate(arrays, axis=0)
        return np.unique(stacked[:, 2])

    def __len__(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return len(self) == 0

    def clear(self) -> None:
        self._queue.clear()

    def next_time(self) -> Optional[TTime]:
        if not self:
            return None
        t, _ = self._queue.peekitem(index=0)
        return t  # type: ignore[return-value]

    # --- Iteration utilities -------------------------------------------------

    def iterate(self) -> Iterator[PopResult[TTime]]:
        while self:
            yield self.pop()

