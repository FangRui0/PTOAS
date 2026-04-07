"""Public type markers for the TileLang DSL v1 surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True)
class ScalarType:
    name: str

    def __repr__(self) -> str:
        return self.name


class TensorView:
    """Bare TensorView annotation marker for TileLang DSL v1."""


class Tile:
    """Bare Tile annotation marker for TileLang DSL v1."""


@dataclass(frozen=True)
class PointerType:
    element_dtype: ScalarType
    memory_space: "MemorySpace"

    def __repr__(self) -> str:
        return f"ptr({self.element_dtype!r}, {self.memory_space!r})"


@dataclass(frozen=True)
class WildcardType:
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeVariable:
    name: str

    def __repr__(self) -> str:
        return f"TypeVar({self.name!r})"


class MemorySpace(str, Enum):
    GM = "gm"
    UB = "ub"


class Pipe(str, Enum):
    MTE1 = "PIPE_MTE1"
    MTE2 = "PIPE_MTE2"
    V = "PIPE_V"
    MTE3 = "PIPE_MTE3"
    ALL = "PIPE_ALL"


class Event(str, Enum):
    ID0 = "EVENT_ID0"
    ID1 = "EVENT_ID1"
    ID2 = "EVENT_ID2"
    ID3 = "EVENT_ID3"
    ID4 = "EVENT_ID4"
    ID5 = "EVENT_ID5"
    ID6 = "EVENT_ID6"
    ID7 = "EVENT_ID7"


class MaskPattern(str, Enum):
    ALL = "PAT_ALL"
    ALLF = "PAT_ALLF"
    EVEN = "PAT_EVEN"
    ODD = "PAT_ODD"
    VL16 = "PAT_VL16"
    VL32 = "PAT_VL32"


@dataclass(frozen=True)
class TileConfig:
    fields: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TileConfig":
        return cls(tuple(sorted(mapping.items())))


@dataclass(frozen=True)
class TileSpecialization:
    shape: tuple[int, ...]
    memory_space: MemorySpace
    config: TileConfig | None = None


i8 = ScalarType("i8")
i1 = ScalarType("i1")
i16 = ScalarType("i16")
i32 = ScalarType("i32")
i64 = ScalarType("i64")
f16 = ScalarType("f16")
bf16 = ScalarType("bf16")
f32 = ScalarType("f32")
PIPE = Pipe
EVENT = Event
PAT = MaskPattern
AnyFloat = WildcardType("AnyFloat")
AnyInt = WildcardType("AnyInt")
AnyType = WildcardType("AnyType")
AnyMask = WildcardType("AnyMask")


def TypeVar(name: str) -> TypeVariable:
    if not isinstance(name, str) or not name:
        raise TypeError("TypeVar name must be a non-empty string")
    return TypeVariable(name)


def ptr(dtype: ScalarType, memory_space: MemorySpace) -> PointerType:
    if not isinstance(dtype, ScalarType):
        raise TypeError("ptr() expects a TileLang scalar dtype")
    if not isinstance(memory_space, MemorySpace):
        raise TypeError("ptr() expects a TileLang MemorySpace")
    return PointerType(element_dtype=dtype, memory_space=memory_space)


def get_lanes(dtype: ScalarType) -> int:
    if not isinstance(dtype, ScalarType):
        raise TypeError("get_lanes expects a TileLang scalar dtype")
    byte_widths = {
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "f16": 2,
        "bf16": 2,
        "f32": 4,
    }
    width = byte_widths.get(dtype.name)
    if width is None:
        raise TypeError(f"dtype `{dtype.name}` is not supported by get_lanes")
    return 256 // width


__all__ = [
    "ScalarType",
    "WildcardType",
    "TypeVariable",
    "TypeVar",
    "TensorView",
    "Tile",
    "PointerType",
    "ptr",
    "MemorySpace",
    "Pipe",
    "Event",
    "PIPE",
    "EVENT",
    "MaskPattern",
    "PAT",
    "TileConfig",
    "TileSpecialization",
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "bf16",
    "f32",
    "AnyFloat",
    "AnyInt",
    "AnyType",
    "AnyMask",
    "get_lanes",
]
