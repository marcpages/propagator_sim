"""Shared typing Protocols for propagator callables."""

from typing import Protocol
import numpy as np
import numpy.typing as npt


class PTimeFn(Protocol):
    """Callable protocol for rate-of-spread time functions.

    Accepts model-specific positional arguments; returns a tuple with
    transition times and rate of spread arrays.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        A pair of arrays (time_minutes, ros_m_per_min).
    """

    def __call__(
        self, *args, **kwargs
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...


class PMoistFn(Protocol):
    """Callable protocol for moisture probability correction functions.

    Parameters
    ----------
    moist : npt.NDArray[np.floating]
        Moisture content array.

    Returns
    -------
    npt.NDArray[np.floating]
        Probability multiplier in [0, 1].
    """

    def __call__(self, moist: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...
