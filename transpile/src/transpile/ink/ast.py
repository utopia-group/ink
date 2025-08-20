from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Any

import transpile.ink.type as t


class BinOpKinds(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    CONCAT = "++"

    GT = ">"
    LT = "<"
    EQ = "="
    AND = "&&"
    OR = "||"


class UnaryOpKinds(Enum):
    NEG = "--"
    NOT = "not"


@dataclass(frozen=True, slots=True, eq=True)
class Expr:
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        raise NotImplementedError


@dataclass(frozen=True, slots=True, eq=True)
class Num(Expr):
    value: int

    def __str__(self):
        return str(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Num", "value": self.value}


@dataclass(frozen=True, slots=True, eq=True)
class Bool(Expr):
    value: bool

    def __str__(self):
        return str(self.value).lower()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Bool", "value": self.value}


@dataclass(frozen=True, slots=True, eq=True)
class Str(Expr):
    value: str

    def __str__(self):
        return f'"{self.value}"'

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Str", "value": self.value}


@dataclass(frozen=True, slots=True, eq=True)
class Var(Expr):
    name: str

    def __str__(self):
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Var", "name": self.name}


@dataclass(frozen=True, slots=True, eq=True)
class UnaryOp(Expr):
    op: UnaryOpKinds
    expr: Expr

    def __str__(self):
        # For simple expressions, use (op expr) format
        # For complex expressions, use op (expr) format
        if self._is_simple_expr(self.expr):
            return f"({self.op.value} {self.expr})"
        else:
            return f"{self.op.value} ({self.expr})"
    
    def _is_simple_expr(self, expr: Expr) -> bool:
        """Check if an expression is simple enough to not need parentheses around the operator"""
        return isinstance(expr, (Var, TupleAccess, Num, Bool, Str))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "UnaryOp", "op": self.op.value, "expr": self.expr.to_dict()}


@dataclass(frozen=True, slots=True, eq=True)
class BinOp(Expr):
    op: BinOpKinds
    left: Expr
    right: Expr

    def __str__(self):
        # Define operator precedence
        precedence = {
            BinOpKinds.OR: 1,
            BinOpKinds.AND: 2,
            BinOpKinds.EQ: 3,
            BinOpKinds.LT: 4,
            BinOpKinds.GT: 4,
            BinOpKinds.CONCAT: 5,
            BinOpKinds.ADD: 6,
            BinOpKinds.SUB: 6,
            BinOpKinds.MUL: 7,
            BinOpKinds.DIV: 7,
        }

        current_prec = precedence.get(self.op, 0)

        # Add parentheses around left operand if its precedence is lower
        left_str = str(self.left)
        if (
            isinstance(self.left, BinOp)
            and precedence.get(self.left.op, 0) < current_prec
        ):
            left_str = f"({left_str})"

        # Add parentheses around right operand if its precedence is lower or equal
        # (for left-associative operators)
        right_str = str(self.right)
        if (
            isinstance(self.right, BinOp)
            and precedence.get(self.right.op, 0) <= current_prec
        ):
            right_str = f"({right_str})"

        return f"{left_str} {self.op.value} {right_str}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "BinOp",
            "op": self.op.value,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass(frozen=True, slots=True, eq=True)
class Ite(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr

    def __str__(self):
        return f"ITE({self.cond}, {self.then_expr}, {self.else_expr})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Ite",
            "cond": self.cond.to_dict(),
            "then_expr": self.then_expr.to_dict(),
            "else_expr": self.else_expr.to_dict(),
        }


@dataclass(frozen=True, slots=True, eq=True)
class App(Expr):
    func: Expr
    arg: Expr

    def __str__(self):
        # Collect all curried arguments to display as multi-argument call
        func, args = self._flatten_curried_call()

        if len(args) == 1:
            return f"{func}({args[0]})"
        else:
            args_str = ", ".join(str(arg) for arg in args)
            return f"{func}({args_str})"

    def _flatten_curried_call(self):
        """Flatten curried function calls into (func, [arg1, arg2, ...])"""
        args = []
        current = self

        # Traverse the chain of Apps to collect all arguments
        while isinstance(current, App):
            args.append(current.arg)
            current = current.func

        # Arguments are in reverse order due to curried structure
        args.reverse()

        return current, args

    def to_dict(self) -> dict[str, Any]:
        return {"type": "App", "func": self.func.to_dict(), "arg": self.arg.to_dict()}


@dataclass(frozen=True, slots=True, eq=True)
class Lambda(Expr):
    param: str
    param_type: t.Type
    body: Expr

    def __str__(self):
        return f"({self.param}: {self.param_type}) -> {self.body}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Lambda",
            "param": self.param,
            "param_type": self.param_type.to_dict(),
            "body": self.body.to_dict(),
        }


@dataclass(frozen=True, slots=True, eq=True)
class Let(Expr):
    name: str
    expr: Expr
    body: Expr

    def __str__(self):
        return f"let {self.name} = {self.expr} in {self.body}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Let",
            "name": self.name,
            "expr": self.expr.to_dict(),
            "body": self.body.to_dict(),
        }


@dataclass(frozen=True, slots=True, eq=True)
class Nil(Expr):
    def __str__(self):
        return "nil"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Nil"}


@dataclass(frozen=True, slots=True, eq=True)
class Cons(Expr):
    head: Expr
    tail: Expr

    def __str__(self):
        return f"{self.head} :: {self.tail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Cons",
            "head": self.head.to_dict(),
            "tail": self.tail.to_dict(),
        }


@dataclass(frozen=True, slots=True, eq=True)
class Tuple(Expr):
    values: tuple[Expr, ...]

    def __str__(self):
        return f"({', '.join(map(str, self.values))})"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Tuple", "values": [value.to_dict() for value in self.values]}


@dataclass(frozen=True, slots=True, eq=True)
class TupleAccess(Expr):
    tuple_expr: Expr
    index: int

    def __str__(self):
        return f"{self.tuple_expr}._{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "TupleAccess",
            "tuple_expr": self.tuple_expr.to_dict(),
            "index": self.index,
        }


@dataclass(frozen=True, slots=True, eq=True)
class MapAccess(Expr):
    map: Expr
    key: Expr

    def __str__(self):
        return f"{self.map}[{self.key}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "MapAccess",
            "map": self.map.to_dict(),
            "key": self.key.to_dict(),
        }


@dataclass(frozen=True, slots=True, eq=True)
class MapAssign(Expr):
    map: Expr
    key: Expr
    value: Expr

    def __str__(self):
        return f"{self.map}[{self.key} <- {self.value}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "MapAssign",
            "map": self.map.to_dict(),
            "key": self.key.to_dict(),
            "value": self.value.to_dict(),
        }


def to_list(elms: list[Expr]) -> Expr:
    if not elms:
        return Nil()
    return reduce(lambda acc, x: Cons(head=x, tail=acc), reversed(elms), Nil())


def bind_lets(lets: list[tuple[str, Expr]], body: Expr) -> Expr:
    if not lets:
        return body
    return reduce(
        lambda acc, let: Let(name=let[0], expr=let[1], body=acc),
        reversed(lets),
        body,
    )


def call_params(func: Expr, args: list[Expr]) -> Expr:
    if not args:
        return func
    return reduce(lambda acc, arg: App(func=acc, arg=arg), args, func)


def call_multi_args(func: Expr, args: list[Expr]) -> Expr:
    """Create a multi-argument function call like func(arg1, arg2, ...)"""
    if not args:
        return func
    elif len(args) == 1:
        return App(func, args[0])
    else:
        # For multiple arguments, pass as a tuple
        return App(func, Tuple(tuple(args)))


def bind_params(body: Expr, params: list[tuple[str, t.Type]]) -> Expr:
    if not params:
        return body
    return reduce(
        lambda acc, param: Lambda(param=param[0], param_type=param[1], body=acc),
        reversed(params),
        body,
    )


def replace_expr(prog: Expr, orig: Expr, to: Expr) -> Expr:
    if prog == orig:
        return to

    match prog:
        case UnaryOp(op, expr):
            return UnaryOp(op, replace_expr(expr, orig, to))
        case BinOp(op, left, right):
            return BinOp(
                op, replace_expr(left, orig, to), replace_expr(right, orig, to)
            )
        case Ite(cond, then_expr, else_expr):
            return Ite(
                replace_expr(cond, orig, to),
                replace_expr(then_expr, orig, to),
                replace_expr(else_expr, orig, to),
            )
        case App(func, arg):
            return App(replace_expr(func, orig, to), replace_expr(arg, orig, to))
        case Lambda(param, param_type, body) if prog != orig:
            return Lambda(param, param_type, replace_expr(body, orig, to))
        case Let(name, expr, body):
            return Let(name, replace_expr(expr, orig, to), replace_expr(body, orig, to))
        case Cons(head, tail):
            return Cons(replace_expr(head, orig, to), replace_expr(tail, orig, to))
        case Tuple(values):
            return Tuple(tuple(replace_expr(v, orig, to) for v in values))
        case TupleAccess(tuple_expr, index):
            return TupleAccess(replace_expr(tuple_expr, orig, to), index)
        case MapAccess(map, key):
            return MapAccess(replace_expr(map, orig, to), replace_expr(key, orig, to))
        case MapAssign(map, key, value):
            return MapAssign(
                replace_expr(map, orig, to),
                replace_expr(key, orig, to),
                replace_expr(value, orig, to),
            )
        case _:
            return prog
