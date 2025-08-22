"""Lightweight event scheduler for propagation updates.

Stores future updates grouped by simulation time and exposes utilities to push
events, pop the earliest batch, and inspect active realizations.
"""

from dataclasses import dataclass, field
from typing import Iterable, List
from sortedcontainers import SortedDict
import numpy as np
import numpy.typing as npt

@dataclass
class Scheduler:
    """Handles scheduling of propagation updates by time."""
    list: SortedDict = field(default_factory=SortedDict)

    def push(self, coords: npt.NDArray[np.integer], time: int) -> None:
        """Schedule a set of coordinates at a given time.

        Parameters
        ----------
        coords : numpy.ndarray
            Array of shape (n, 3) with [row, col, realization].
        time : int | float
            Simulation time when these updates occur.
        """
        if time not in self.list:
            self.list[time] = []
        self.list[time].append(coords)

    def push_all(self, updates: Iterable[tuple[int, npt.NDArray[np.integer]]]) -> None:
        """Push multiple updates.

        Parameters
        ----------
        updates : Iterable[tuple[int, numpy.ndarray]]
            Pairs of (time, coords array).
        """
        for t, u in updates:
            self.push(u, t)

    def pop(self) -> tuple[int, List[npt.NDArray[np.integer]]]:
        """Pop and return the earliest scheduled batch.

        Returns
        -------
        tuple[int, list[numpy.ndarray]]
            The time and list of coord arrays.
        """
        item = self.list.popitem(index=0)
        return item

    def active(self) -> npt.NDArray[np.integer]:
        """
        Return the active realization indices that have a scheduled update.

        Returns
        -------
        numpy.ndarray
            1D array of unique realization indices.
        """
        active_t = np.unique(
            [e for k in self.list.keys() for c in self.list[k] for e in c[:, 2]]
        )
        return active_t

    def __len__(self) -> int:
        """Number of distinct scheduled time keys."""
        return len(self.list)

    def next_time(self) -> int | None:
        """
        Return the earliest scheduled time without mutating the queue.

        Returns
        -------
        int | None
            Earliest time or None if empty.
        """
        if len(self) == 0:
            return None

        next_time, _next_updates = self.list.peekitem(index=0)
        return next_time # type: ignore

    def __call__(self):
        """Iterate over scheduled updates, allowing dynamic rescheduling."""
        while len(self) > 0:
            c_time, updates = self.pop()
            print("u")
            new_updates: Iterable[tuple[int, npt.NDArray[np.integer]]] = yield c_time, updates
            print("n")
            self.push_all(new_updates)
