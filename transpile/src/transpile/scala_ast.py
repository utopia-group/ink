from dataclasses import dataclass, field, fields
from functools import singledispatch
from itertools import groupby
from typing import Any

import tree_sitter_scala as tss
from tree_sitter import Language, Node, Parser, Tree

MATCHED_BRACKETS = {"{": "}", "(": ")", "[": "]"}
BOOLEAN_LITERALS = {"true": True, "false": False}


@dataclass(frozen=True, slots=True, eq=True)
class ScalaNode:
    node: Node = field(repr=False)


@dataclass(frozen=True, slots=True, eq=True)
class Var(ScalaNode):
    name: str


@dataclass(frozen=True, slots=True, eq=True)
class Constant(ScalaNode):
    value: int | str | bool


@singledispatch
def parse_tree(arg) -> ScalaNode | list[ScalaNode] | str:
    raise NotImplementedError(f"parse_tree not implemented for {type(arg)}: {arg}")


@parse_tree.register
def _(arg: type(None)) -> None:
    return None


@parse_tree.register
def _(arg: bytes) -> str:
    return arg.decode("utf-8")


@dataclass(frozen=True, slots=True, eq=True)
class Type(ScalaNode):
    name: str


@dataclass(frozen=True, slots=True, eq=True)
class TupleType(Type):
    elms: list[Type]


@dataclass(frozen=True, slots=True, eq=True)
class GenericType(Type):
    args: list[Type]


@dataclass(frozen=True, slots=True, eq=True)
class ParameterDecl(ScalaNode):
    name: str
    type: Type


@dataclass(frozen=True, slots=True, eq=True)
class FunctionDecl(ScalaNode):
    name: str
    params: list[ParameterDecl]
    type: Type
    body: list[ScalaNode]


@dataclass(frozen=True, slots=True, eq=True)
class Call(ScalaNode):
    func: ScalaNode
    args: list[ScalaNode]


@dataclass(frozen=True, slots=True, eq=True)
class Assignment(ScalaNode):
    target: ScalaNode
    value: ScalaNode


@dataclass(frozen=True, slots=True, eq=True)
class ClassDecl(ScalaNode):
    name: str
    extends: str | None
    body: list[ScalaNode]


@dataclass(frozen=True, slots=True, eq=True)
class CaseClassDecl(ScalaNode):
    name: str
    params: list[ParameterDecl]


@dataclass(frozen=True, slots=True, eq=True)
class ModuleDecl(ScalaNode):
    body: list[ScalaNode]


@dataclass(frozen=True, slots=True, eq=True)
class If(ScalaNode):
    test: ScalaNode
    body: list[ScalaNode]
    orelse: list[ScalaNode]


@dataclass(frozen=True, slots=True, eq=True)
class VarDecl(ScalaNode):
    name: str
    type: Type | None
    value: ScalaNode


@dataclass(frozen=True, slots=True, eq=True)
class Field(ScalaNode):
    expr: ScalaNode
    field: str


@dataclass(frozen=True, slots=True, eq=True)
class GenericFunction(ScalaNode):
    expr: ScalaNode
    types: list[Type]


@dataclass(frozen=True, slots=True, eq=True)
class PrefixExpr(ScalaNode):
    op: str
    expr: ScalaNode


@dataclass(frozen=True, slots=True, eq=True)
class InfixExpr(ScalaNode):
    left: ScalaNode
    op: str
    right: ScalaNode


@dataclass(frozen=True, slots=True, eq=True)
class TupleExpr(ScalaNode):
    elms: list[ScalaNode]


@dataclass(frozen=True, slots=True, eq=True)
class Throw(ScalaNode):
    expr: ScalaNode


@dataclass(frozen=True, slots=True, eq=True)
class InstanceExpr(ScalaNode):
    type_id: str
    args: list[ScalaNode]


@dataclass(frozen=True, slots=True, eq=True)
class LambdaExpr(ScalaNode):
    param: ParameterDecl
    body: ScalaNode


def index_of_type(nodes: list[Node], type: str | tuple[str, ...]) -> int:
    types = type if isinstance(type, tuple) else (type,)
    for i, node in enumerate(nodes):
        if node.type in types:
            return i
    raise ValueError(f"type {type} not found in {nodes}")


def parse_positional_type_annotation(
    children: list[Node], separator: str = ":"
) -> Type:
    colon_idx = index_of_type(children, separator)
    return parse_tree(children[colon_idx + 1])


def filter_comment(nodes: list[Node]) -> list[Node]:
    return [child for child in nodes if child.type not in ["comment", "block_comment"]]


def find_all(nodes: list[Node], type: str | tuple[str, ...]) -> list[Node]:
    types = type if isinstance(type, tuple) else (type,)
    return [child for child in nodes if any(child.type == t for t in types)]


def find_single(nodes: list[Node], type: str | tuple[str, ...]) -> Node:
    children = find_all(nodes, type)
    assert len(children) == 1, f"expected 1 {type} found {len(children)}"
    return children[0]


def ensure_var(arg: ScalaNode | str | list) -> ScalaNode:
    if isinstance(arg, str):
        return Var(node=None, name=arg)
    elif isinstance(arg, list):
        # If it's a list, take the first non-None element
        # This handles cases where indented_block returns multiple statements
        for item in arg:
            if item is not None:
                return ensure_var(item)
        # If no non-None items, return a placeholder
        return Var(node=None, name="PLACEHOLDER")
    assert isinstance(arg, ScalaNode)
    return arg


def ensure_stmt_list(arg: Any | list[Any]) -> list[Any]:
    if arg is None:
        return []
    args = arg if isinstance(arg, list) else [arg]
    return list(map(ensure_var, args))


def check_brackets(children: list[Node], optional=False) -> list[Node]:
    matched_first = children[0].type in MATCHED_BRACKETS
    if not matched_first and not optional:
        raise AssertionError("first child should be bracket")

    matched_last = children[-1].type == MATCHED_BRACKETS[children[0].type]
    assert matched_last == matched_first, "last child should be matching bracket"
    return children[1:-1] if matched_first else children


def split_by_type(nodes: list[Node], type: str) -> list[list[Node]]:
    """
    split the nodes where the first type node is used as the delimiter
    the delimiter node is not included in the result
    """
    return [
        list(group) for _, group in groupby(nodes, lambda x: x.type == type) if not _
    ]


@parse_tree.register
def _(arg: Tree) -> ScalaNode | str | None:
    return parse_tree(arg.root_node)


@parse_tree.register
def _(arg: Node) -> ScalaNode | str | None:
    children = filter_comment(arg.children)
    named_children = filter_comment(arg.named_children)
    match arg.type:
        case "var_definition" | "package_clause" | "import_declaration":
            return None

        case "comment" | "block_comment":
            return None

        case "wildcard":
            return Var(node=arg, name=parse_tree(arg.text))

        case "interpolated_string_expression":
            return Constant(node=arg, value=parse_tree(arg.text).strip('s"'))

        case "compilation_unit":
            return ModuleDecl(
                node=arg, body=list(filter(None, map(parse_tree, children)))
            )

        case "floating_point_literal":
            return Constant(node=arg, value=float(parse_tree(arg.text).strip("dD")))

        case "integer_literal":
            return Constant(node=arg, value=int(parse_tree(arg.text).strip("L")))

        case "boolean_literal":
            return Constant(node=arg, value=BOOLEAN_LITERALS[parse_tree(arg.text)])

        case "string":
            return Constant(node=arg, value=parse_tree(arg.text).strip('"'))

        case "identifier":
            return parse_tree(arg.text)

        case "operator_identifier":
            return parse_tree(arg.text)

        case "type_identifier" | "stable_type_identifier":
            return Type(node=arg, name=parse_tree(arg.text))

        case "extends_clause":
            return parse_positional_type_annotation(children, "extends")

        case "type_arguments":
            return [parse_tree(x) for x in named_children]

        case "tuple_type":
            return TupleType(
                node=arg,
                name="Tuple",
                elms=[parse_tree(x) for x in named_children],
            )

        case "tuple_expression":
            return TupleExpr(
                node=arg,
                elms=[parse_tree(x) for x in named_children],
            )

        case "field_expression":
            return Field(
                node=arg,
                expr=ensure_var(parse_tree(named_children[0])),
                field=parse_tree(named_children[1]),
            )

        case "generic_function":
            return GenericFunction(
                node=arg,
                expr=parse_tree(named_children[0]),
                types=parse_tree(named_children[1]),
            )

        case "generic_type":
            return GenericType(
                node=arg,
                name=parse_tree(named_children[0]),
                args=parse_tree(named_children[1]),
            )
        case "class_parameters":
            return [parse_tree(x) for x in find_all(children, "class_parameter")]

        case "class_parameter":
            name = parse_tree(find_single(children, "identifier"))
            type = parse_positional_type_annotation(children)
            return ParameterDecl(node=arg, name=name, type=type)

        case "class_definition" | "object_definition" if children[0].type == "case":
            name = parse_tree(children[2])
            params = parse_tree(find_single(children, "class_parameters"))

            return CaseClassDecl(
                node=arg,
                name=name,
                params=params,
            )

        case "class_definition" | "object_definition":
            name = parse_tree(children[1])
            extends_clause_node = next(
                filter(lambda x: x.type == "extends_clause", children), None
            )
            extends_clause = (
                parse_tree(extends_clause_node) if extends_clause_node else None
            )

            body = check_brackets(find_single(children, "template_body").children)
            body = list(filter(None, map(parse_tree, body)))
            return ClassDecl(node=arg, name=name, extends=extends_clause, body=body)

        case "parameter":
            name = parse_tree(named_children[0])
            type = parse_tree(named_children[1])
            return ParameterDecl(node=arg, name=name, type=type)

        case "block":
            return list(filter(None, map(parse_tree, check_brackets(children))))

        case "function_definition":
            [decl, body] = split_by_type(children, "=")
            name = parse_tree(find_single(decl, "identifier"))
            params = None
            if any(x.type == "parameters" for x in decl):
                params = list(
                    map(
                        parse_tree,
                        find_all(
                            find_single(children, "parameters").children, "parameter"
                        ),
                    )
                )

            assert len(body) == 1, "function_definition should have 1 body"
            return FunctionDecl(
                node=arg,
                name=name,
                params=params,
                type=parse_positional_type_annotation(children),
                body=ensure_stmt_list(parse_tree(body[0])),
            )

        case "call_expression":
            assert len(children) == 2, "call_expression should have 2 children"
            func = parse_tree(children[0])
            args_node = children[1]
            if args_node.type == "case_block":
                # TODO: handle case_block
                return Var(node=arg, name="UNSUPPORTED_case_block")
            assert args_node.type == "arguments", (
                "call_expression second child should be arguments"
            )
            return Call(
                node=arg,
                func=func,
                args=[
                    ensure_var(parse_tree(c))
                    for c in filter_comment(args_node.named_children)
                ],
            )

        case "assignment_expression":
            assert len(children) == 3 and children[1].type == "="
            target = parse_tree(children[0])
            value = ensure_var(parse_tree(children[2]))
            return Assignment(node=arg, target=target, value=value)

        case "parenthesized_expression":
            assert arg.named_child_count == 1, (
                "parenthesized_expression should have 1 child"
            )
            return parse_tree(arg.named_children[0])

        case "if_expression":
            test = ensure_var(parse_tree(arg.named_children[0]))
            body = parse_tree(arg.named_children[1])
            orelse = (
                parse_tree(arg.named_children[2])
                if len(arg.named_children) == 3
                else None
            )
            return If(
                node=arg,
                test=test,
                body=ensure_stmt_list(body),
                orelse=ensure_stmt_list(orelse),
            )

        case "prefix_expression":
            return PrefixExpr(
                node=arg,
                op=parse_tree(children[0].text),
                expr=ensure_var(parse_tree(children[1])),
            )

        case "infix_expression":
            return InfixExpr(
                node=arg,
                left=ensure_var(parse_tree(children[0])),
                op=parse_tree(children[1]),
                right=ensure_var(parse_tree(children[2])),
            )

        case "val_definition":
            if arg.named_child_count == 2:
                return VarDecl(
                    node=arg,
                    name=parse_tree(arg.named_children[0]),
                    type=None,
                    value=ensure_var(parse_tree(arg.named_children[1])),
                )
            else:
                assert arg.named_child_count == 3
                return VarDecl(
                    node=arg,
                    name=parse_tree(arg.named_children[0]),
                    type=parse_tree(arg.named_children[1]),
                    value=ensure_var(parse_tree(arg.named_children[2])),
                )

        case "indented_block":
            if len(named_children) == 1:
                return parse_tree(named_children[0])
            else:
                return list(filter(None, map(parse_tree, named_children)))

        case "throw_expression":
            return Throw(node=arg, expr=parse_tree(children[1]))

        case "instance_expression":
            args = named_children[1].named_children if len(named_children) > 1 else []
            return InstanceExpr(
                node=arg,
                type_id=parse_tree(named_children[0]),
                args=[ensure_var(parse_tree(c)) for c in filter_comment(args)],
            )

        case "lambda_expression":
            param, body = split_by_type(children, "=>")
            assert len(param) == 1, "lambda_expression should have 1 parameter"
            assert len(body) == 1, "lambda_expression should have 1 body"
            return LambdaExpr(
                node=arg,
                param=parse_tree(param[0]),
                body=ensure_var(parse_tree(body[0])),
            )

        case "modifiers":
            # Ignore modifiers for now (like private, public, etc.)
            return None

        case "postfix_expression":
            # Handle method calls like .toString, .toInt, etc.
            expr = parse_tree(children[0])
            method = parse_tree(children[1])
            return Field(node=arg, expr=ensure_var(expr), field=method)

        case "match_expression":
            # For now, just return a placeholder for match expressions
            return Var(node=arg, name="UNSUPPORTED_match_expression")

        case _:
            raise NotImplementedError(arg.type)


class NodeVisitor[Ret]:
    def visit(self, node: ScalaNode) -> Ret:
        """Visit a node."""
        method_name = "visit_" + node.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Node | str) -> Ret:
        """Called if no explicit visitor function exists for a node."""
        for f in fields(node):
            value = getattr(node, f.name)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ScalaNode):
                        self.visit(item)
            elif isinstance(value, ScalaNode):
                self.visit(value)


def parse(src: str) -> ScalaNode:
    SCALA_LANGUAGE = Language(tss.language())

    parser = Parser(SCALA_LANGUAGE)
    tree = parser.parse(src.encode("utf-8"))
    return parse_tree(tree.root_node)
