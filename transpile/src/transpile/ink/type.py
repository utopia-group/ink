from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, eq=True)
class Type:
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        raise NotImplementedError


@dataclass(frozen=True, slots=True, eq=True)
class Num(Type):
    def __str__(self):
        return "num"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Num"}


@dataclass(frozen=True, slots=True, eq=True)
class Bool(Type):
    def __str__(self):
        return "bool"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Bool"}


@dataclass(frozen=True, slots=True, eq=True)
class Str(Type):
    def __str__(self):
        return "str"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Str"}


@dataclass(frozen=True, slots=True, eq=True)
class List(Type):
    elm: Type

    def __str__(self):
        return f"list[{self.elm}]"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "List", "elm": self.elm.to_dict()}


@dataclass(frozen=True, slots=True, eq=True)
class Map(Type):
    key: Type
    value: Type

    def __str__(self):
        return f"map[{self.key}, {self.value}]"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Map", "key": self.key.to_dict(), "value": self.value.to_dict()}


@dataclass(frozen=True, slots=True, eq=True)
class Set(Type):
    elm: Type

    def __str__(self):
        return f"set[{self.elm}]"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Set", "elm": self.elm.to_dict()}


@dataclass(frozen=True, slots=True, eq=True)
class Tuple(Type):
    elms: tuple[Type, ...]

    def __str__(self):
        return f"({', '.join(map(str, self.elms))})"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Tuple", "elms": [elm.to_dict() for elm in self.elms]}


@dataclass(frozen=True, slots=True, eq=True)
class Fn(Type):
    param: Type
    ret: Type

    def __str__(self):
        return f"({self.param} -> {self.ret})"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Fn", "param": self.param.to_dict(), "ret": self.ret.to_dict()}
