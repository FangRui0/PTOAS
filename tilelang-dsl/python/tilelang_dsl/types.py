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
i16 = ScalarType("i16")
i32 = ScalarType("i32")
i64 = ScalarType("i64")
f16 = ScalarType("f16")
bf16 = ScalarType("bf16")
f32 = ScalarType("f32")
AnyFloat = WildcardType("AnyFloat")
AnyInt = WildcardType("AnyInt")
AnyType = WildcardType("AnyType")
AnyMask = WildcardType("AnyMask")


def TypeVar(name: str) -> TypeVariable:
    if not isinstance(name, str) or not name:
        raise TypeError("TypeVar name must be a non-empty string")
    return TypeVariable(name)


__all__ = [
    "ScalarType",
    "WildcardType",
    "TypeVariable",
    "TypeVar",
    "TensorView",
    "Tile",
    "MemorySpace",
    "TileConfig",
    "TileSpecialization",
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
]
