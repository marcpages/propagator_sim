"""Lightweight event scheduler for propagation updates.

Stores future updates grouped by simulation time and exposes utilities to push
events, pop the earliest batch, and inspect active realizations.
"""

from sortedcontainers import SortedDict
import numpy as np


class Scheduler:
    """Handles scheduling of propagation updates by time."""

    def __init__(self):
        self.list = SortedDict()

        # fix the change in SortedDict api
        self.list_kw = {"last": False}
        try:
            self.list.popitem(**self.list_kw)
        except KeyError:
            pass
        except TypeError:
            self.list_kw = {"index": 0}

    def push(self, coords, time):
        """Schedule a set of coordinates at a given time.

        Args:
            coords (np.ndarray): Array of shape (n, 3) with [row, col, realization].
            time (int | float): Simulation time when these updates occur.
        """
        if time not in self.list:
            self.list[time] = []
        self.list[time].append(coords)

    def push_all(self, updates):
        """Push multiple updates.

        Args:
            updates (list[tuple[int | float, np.ndarray]]): Pairs of (time, coords).
        """
        for t, u in updates:
            self.push(u, t)

    def pop(self):
        """Pop and return the earliest scheduled batch.

        Returns:
            tuple[int | float, list[np.ndarray]]: The time and list of coord arrays.
        """
        item = self.list.popitem(**self.list_kw)
        return item

    def active(self):
        """
        Return the active realization indices that have a scheduled update.

        Returns:
            np.ndarray: 1D array of unique realization indices.
        """
        active_t = np.unique(
            [e for k in self.list.keys() for c in self.list[k] for e in c[:, 2]]
        )
        return active_t

    def __len__(self):
        """Number of distinct scheduled time keys."""
        return len(self.list)

    def next_time(self) -> int | None:
        """
        Return the earliest scheduled time without mutating the queue.

        Returns:
            int | float | None: Earliest time or None if empty.
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
            new_updates = yield c_time, updates
            print("n")
            self.push_all(new_updates)
