from __future__ import annotations

from typing import Any, Iterable, Type, cast, List, Tuple, Literal
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from functools import lru_cache

from propagator_io.geometry import Geometry, GeometryParser
from propagator_io.geo import GeographicInfo
from propagator_io.geometry import GeometryKind, rasterize_geometries


class ActionType(str, Enum):
    WATERLINE_ACTION = "waterline_action"
    CANADAIR = "canadair"
    HELICOPTER = "helicopter"
    HEAVY_ACTION = "heavy_action"


# ---------- Base class ----------
class Action(BaseModel):

    geometries: List[Geometry] = Field(default_factory=list)

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return set()

    @field_validator("geometries")
    @classmethod
    def _check_allowed(cls, geoms: List[Geometry]) -> List[Geometry]:
        allowed = cls.allowed_kinds()
        for g in geoms:
            if g.kind not in allowed:
                raise ValueError(f"{cls.__name__} supports {allowed},\
                    got {g.kind}")
        return geoms

    def _mask(self, geo_info: GeographicInfo) -> np.ndarray:
        m = rasterize_geometries(
            geometries=self.geometries,
            geo_info=geo_info,
            # fill: int = 0,
            # default_value: Union[int, float] = 1,
            # values: Optional[Sequence[Union[int, float]]] = None,
            # all_touched: bool = True,
            # dtype: str = "uint8",
        )
        return m.astype(bool)

    def apply(
        self,
        geo_info: GeographicInfo,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        raise NotImplementedError()


# ---------- Concrete actions ----------

class WaterlineAction(Action):
    action_type: Literal[ActionType.WATERLINE_ACTION] = Field(
        default=ActionType.WATERLINE_ACTION, frozen=True
    )

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return {GeometryKind.LINE}

    def apply(
        self,
        geo_info: GeographicInfo,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return (np.zeros(geo_info.shape, dtype=float),
                np.zeros(geo_info.shape, dtype=float))


class CanadairAction(Action):
    action_type: Literal[ActionType.CANADAIR] = Field(
        default=ActionType.CANADAIR, frozen=True)

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return {GeometryKind.LINE}

    def apply(
        self,
        geo_info: GeographicInfo,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return (np.zeros(geo_info.shape, dtype=float),
                np.zeros(geo_info.shape, dtype=float))


class HelicopterAction(Action):
    action_type: Literal[ActionType.HELICOPTER] = Field(
        default=ActionType.HELICOPTER, frozen=True)

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return {GeometryKind.LINE}

    def apply(
        self,
        geo_info: GeographicInfo,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return (np.zeros(geo_info.shape, dtype=float),
                np.zeros(geo_info.shape, dtype=float))


class HeavyAction(Action):
    action_type: Literal[ActionType.HEAVY_ACTION] = Field(
        default=ActionType.HEAVY_ACTION, frozen=True)

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return {GeometryKind.LINE}

    def apply(
        self,
        geo_info: GeographicInfo,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return (np.zeros(geo_info.shape, dtype=float),
                np.zeros(geo_info.shape, dtype=float))


# ---------- parsing for boundary conditions definition ----------


def _iter_subclasses(cls: Type[Action]) -> Iterable[Type[Action]]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


@lru_cache(maxsize=1)
def get_action_registry() -> dict[ActionType, Type[Action]]:
    """
    Build once by introspecting subclasses (no manual lists).
    Pydantic v2 stores field metadata on `model_fields`.
    """
    reg: dict[ActionType, Type[Action]] = {}
    for sub in _iter_subclasses(Action):
        # Be defensive: model_fields may not exist on unrelated classes
        fields: dict[str, Any] = cast(dict[str, Any],
                                      getattr(sub, "model_fields", {}))
        info = fields.get("action_type")
        default = getattr(info, "default", None)
        if isinstance(default, ActionType):
            reg[default] = sub
    return reg


@lru_cache(maxsize=1)
def _action_name_set() -> frozenset[str]:
    return frozenset(a.value for a in ActionType)


def load_action(obj: dict[str, Any]) -> Action:
    """
    Instantiate the right Action subclass from a dict containing 'action_type'.
    Accepts str or ActionType.
    """
    atype_raw = obj.get("action_type")
    atype = ActionType(atype_raw) if isinstance(atype_raw, str) \
        else cast(ActionType, atype_raw)
    cls = get_action_registry().get(atype)
    if cls is None:
        raise ValueError(f"Unknown action_type: {atype_raw!r}")
    return cls.model_validate(obj)


def parse_actions(
    data: dict[str, Any],
    epsg: int,
) -> tuple[list[Action], set[str]]:
    reg = get_action_registry()
    valid_names = _action_name_set()
    actions: list[Action] = []
    consumed: set[str] = set()
    # Iterate over a copy since we may remove keys
    for key, raw in list(data.items()):
        # Is this key the name of an action?
        if key not in valid_names or not raw:
            continue
        atype = ActionType(key)
        cls = reg[atype]
        if cls is None:
            continue
        allowed = {k.value for k in cls.allowed_kinds()}
        if isinstance(raw, list) and (not raw or
                                      isinstance(raw[0], (str, dict))):
            geoms = GeometryParser.parse_geometry_list(
                raw, allowed=allowed, epsg=epsg)
        else:
            geoms = raw  # assume already parsed to Geometry objects
        actions.append(cls(geometries=geoms))
        consumed.add(key)
    return actions, consumed
