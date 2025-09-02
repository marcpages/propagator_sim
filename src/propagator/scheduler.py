"""Lightweight event scheduler for propagation updates.

Stores future updates grouped by simulation time and exposes utilities to push
events, pop the earliest batch, and inspect active realizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from propagator.models import BoundaryConditions, CoordsTuple

PopResult = Tuple[int, "SchedulerEvent"]


@dataclass
class SchedulerEvent:
    """Represents a scheduled event in the simulation."""

    coords: List[CoordsTuple] = field(default_factory=list)

    # boundary_conditions
    moisture: Optional[npt.NDArray[np.floating]] = None
    wind_dir: Optional[npt.NDArray[np.floating]] = None
    wind_speed: Optional[npt.NDArray[np.floating]] = None

    # actions
    additional_moisture: Optional[npt.NDArray[np.floating]] = None
    vegetation_changes: Optional[npt.NDArray[np.floating]] = None


@dataclass(frozen=True)
class SortedDict:
    """Represents a sorted dictionary for scheduling events."""

    _data: Dict[int, SchedulerEvent] = field(
        default_factory=dict, init=False, repr=False
    )
    _order: List[int] = field(default_factory=list, init=False, repr=False)

    def __setitem__(self, key: int, value: SchedulerEvent) -> None:
        self._data[key] = value
        self._order.append(key)
        self._order.sort()

    def __getitem__(self, key: int) -> SchedulerEvent:
        return self._data[key]

    def __delitem__(self, key: int) -> None:
        del self._data[key]
        self._order.remove(key)

    def __iter__(self) -> Iterator[int]:
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._data)

    def get(
        self, key: int, default: Optional[SchedulerEvent] = None
    ) -> Optional[SchedulerEvent]:
        return self._data.get(key, default)

    def popitem(self, index: int) -> tuple[int, SchedulerEvent]:
        key = self._order.pop(index)
        value = self._data.pop(key)
        return key, value

    def values(self) -> Iterator[SchedulerEvent]:
        return iter(self._data.values())

    def items(self) -> Iterator[Tuple[int, SchedulerEvent]]:
        for key in self._order:
            yield key, self._data[key]

    def clear(self) -> None:
        self._data.clear()
        self._order.clear()

    def peekitem(self, index: int) -> Tuple[int, SchedulerEvent]:
        key = self._order[index]
        value = self._data[key]
        return key, value


@dataclass
class Scheduler:
    """
    Lightweight event scheduler for propagation updates.

    Generic over the time key type (int or float), so your inputs and outputs
    stay consistent.
    """

    realizations: int
    _queue: SortedDict = field(
        default_factory=SortedDict, init=False, repr=False
    )

    # --- Basic queue ops -----------------------------------------------------

    def push_ignition(self, time: int, coords: CoordsTuple) -> None:
        event: SchedulerEvent | None
        if time in self._queue:
            event = self._queue.get(time)
        else:
            event = SchedulerEvent()
            self._queue[time] = event
        if event is None:
            raise ValueError("SchedulerEvent should not be None here")
        event.coords.append(coords)

    def pop(self) -> PopResult:
        if not self:
            raise IndexError("pop from empty Scheduler")
        time, updates = self._queue.popitem(index=0)
        return time, updates

    def add_boundary_conditions(self, boundary_conditions: BoundaryConditions):
        """
        Adds a boundary condition to the scheduler.

        Parameters
        ----------
        boundary_conditions : PropagatorBoundaryConditions
            The boundary condition to add at defined time.
        """
        entry = self._queue.get(boundary_conditions.time, None)
        if entry is None:
            entry = SchedulerEvent()
            self._queue[boundary_conditions.time] = entry
        if boundary_conditions.moisture is not None:
            entry.moisture = boundary_conditions.moisture
        if boundary_conditions.wind_dir is not None:
            entry.wind_dir = boundary_conditions.wind_dir
        if boundary_conditions.wind_speed is not None:
            entry.wind_speed = boundary_conditions.wind_speed

        if boundary_conditions.ignition_mask is not None:
            ign_arr = boundary_conditions.ignition_mask
            points = np.argwhere(ign_arr > 0)
            for point in points:
                for realization in range(self.realizations):
                    coords = (point[0], point[1], realization)
                    self.push_ignition(
                        time=boundary_conditions.time, coords=coords
                    )

        if boundary_conditions.additional_moisture is not None:
            if entry.additional_moisture is None:
                entry.additional_moisture = (
                    boundary_conditions.additional_moisture
                )
            else:
                entry.additional_moisture += (
                    boundary_conditions.additional_moisture
                )
        if boundary_conditions.vegetation_changes is not None:
            if entry.vegetation_changes is None:
                entry.vegetation_changes = (
                    boundary_conditions.vegetation_changes
                )
            else:
                entry.vegetation_changes = np.where(
                    boundary_conditions.vegetation_changes == 0,
                    entry.vegetation_changes,
                    boundary_conditions.vegetation_changes,
                )

    def active(self) -> npt.NDArray[np.integer]:
        if not self:
            return np.array([], dtype=int)
        arrays = [
            a for batches in self._queue.values() for a in batches.coords
        ]
        if len(arrays) == 1:
            the_coords = arrays[0]
            return np.array([the_coords[2]])

        stacked = np.vstack(arrays)
        return np.unique(stacked[:, 2])

    def __len__(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return len(self) == 0

    def clear(self) -> None:
        self._queue.clear()

    def next_time(self) -> Optional[int]:
        if not self:
            return None
        t, _ = self._queue.peekitem(index=0)
        return t  # type: ignore[return-value]

    # --- Iteration utilities -------------------------------------------------

    def iterate(self) -> Iterator[PopResult]:
        while self:
            yield self.pop()
