"""
Transpiler for Spark Aggregator to ink programs.

This module handles conversion of Spark Aggregator implementations to ink programs,
focusing specifically on the reduce method.
"""

import json
from typing import Any

import transpile.ink.ast as ink
import transpile.ink.type as t
import transpile.scala_ast as s


def transpile(scala_code: str) -> tuple[ink.Expr, ink.Expr]:
    ast = s.parse(scala_code)
    transpiler = AggregatorTranspiler()
    return transpiler.transpile(ast)


class AggregatorTranspiler:
    """Transpiler for Spark Aggregator to ink programs."""

    def __init__(self):
        self.buffer_param_name = None
        self.input_param_name = None
        self.buffer_type = None
        self.input_type = None
        self.input_case_class = None
        self.buffer_case_class = None
        self.input_field_map = {}
        self.buffer_field_map = {}
        self.ast = None  # Store AST for case class lookups

    def transpile(self, ast: s.ModuleDecl) -> tuple[ink.Expr, ink.Expr]:
        """
        Transpile a parsed Scala AST to ink expressions.

        Args:
            ast: Parsed Scala AST

        Returns:
            Tuple of (reduce_function, zero_value) ink expressions
        """
        # Store AST for case class lookups
        self.ast = ast

        # Find the Aggregator class
        aggregator_class = self._find_aggregator_class(ast)

        # Extract type information from the class declaration
        self._extract_types(aggregator_class, ast)

        # Find the reduce method
        reduce_method = self._find_reduce_method(aggregator_class)

        # Convert the reduce method to ink
        reduce_body = self._convert_reduce_method(reduce_method)

        # Build the reduce function: (buffer: buffer_type) -> (input: input_type) -> body
        reduce_function = ink.Lambda(
            param=self.buffer_param_name,
            param_type=self.buffer_type,
            body=ink.Lambda(
                param=self.input_param_name,
                param_type=self.input_type,
                body=reduce_body,
            ),
        )

        # Find and convert the zero method
        zero_method = self._find_zero_method(aggregator_class)
        zero_value = self._convert_zero_method(zero_method)

        return (reduce_function, zero_value)

    def _find_aggregator_class(self, ast: s.ModuleDecl) -> s.ClassDecl:
        """Find the class that extends Aggregator."""

        def search_classes(nodes):
            for node in nodes:
                if isinstance(node, s.ClassDecl):
                    if self._is_aggregator_class(node):
                        return node
                    # Search nested classes
                    nested_result = search_classes(node.body)
                    if nested_result:
                        return nested_result
            return None

        result = search_classes(ast.body)
        if result:
            return result
        raise ValueError("No Aggregator class found in the AST")

    def _is_aggregator_class(self, class_decl: s.ClassDecl) -> bool:
        """Check if a class extends Aggregator."""
        if not class_decl.extends:
            return False

        if isinstance(class_decl.extends, s.GenericType):
            return class_decl.extends.name.name == "Aggregator"
        elif isinstance(class_decl.extends, s.Type):
            return class_decl.extends.name == "Aggregator"

        return False

    def _extract_types(self, aggregator_class: s.ClassDecl, ast: s.ModuleDecl):
        """Extract type information from the aggregator class and case classes."""

        def _build_field_map(case_class: s.CaseClassDecl) -> dict[str, int]:
            """Build a mapping from field names to tuple indices for a case class."""
            return {param.name: i for i, param in enumerate(case_class.params)}

        def _find_case_class(ast: s.ModuleDecl, name: str) -> s.CaseClassDecl | None:
            """Find a case class by name, return None if not found."""
            for node in ast.body:
                if isinstance(node, s.CaseClassDecl) and node.name == name:
                    return node
            return None

        # Get generic type arguments from Aggregator[Input, Buffer, Output]
        if isinstance(aggregator_class.extends, s.GenericType):
            type_args = aggregator_class.extends.args
            if len(type_args) >= 2:
                input_type_name = type_args[0].name
                buffer_type_name = type_args[1].name

                # Try to find case classes, but handle primitive types
                self.input_case_class = _find_case_class(ast, input_type_name)
                self.buffer_case_class = _find_case_class(ast, buffer_type_name)

                # Build field maps only for case classes
                self.input_field_map = (
                    _build_field_map(self.input_case_class)
                    if self.input_case_class
                    else {}
                )
                self.buffer_field_map = (
                    _build_field_map(self.buffer_case_class)
                    if self.buffer_case_class
                    else {}
                )

                # Convert to ink types
                if self.input_case_class:
                    self.input_type = self._case_class_to_ink_type(
                        self.input_case_class
                    )
                else:
                    # Handle primitive type
                    self.input_type = self._scala_type_to_ink_type(type_args[0])

                if self.buffer_case_class:
                    self.buffer_type = self._case_class_to_ink_type(
                        self.buffer_case_class
                    )
                else:
                    # Handle primitive type
                    self.buffer_type = self._scala_type_to_ink_type(type_args[1])

    def _case_class_to_ink_type(self, case_class: s.CaseClassDecl) -> t.Type:
        """Convert a case class to an ink type."""
        if len(case_class.params) == 1:
            return self._scala_type_to_ink_type(case_class.params[0].type)
        else:
            # Multiple parameters become a tuple
            param_types = [
                self._scala_type_to_ink_type(p.type) for p in case_class.params
            ]
            return t.Tuple(tuple(param_types))

    def _find_case_class_by_name(self, name: str) -> s.CaseClassDecl | None:
        """Find a case class by name in the AST."""
        if not self.ast:
            return None
        for node in self.ast.body:
            if isinstance(node, s.CaseClassDecl) and node.name == name:
                return node
        return None

    def _scala_type_to_ink_type(self, scala_type: s.Type) -> t.Type:
        """Convert a Scala type to an ink type."""

        if isinstance(scala_type, s.GenericType):
            match scala_type.name.name:
                case "Map":
                    # Map[K, V] -> Map(K, V)
                    if len(scala_type.args) == 2:
                        key_type = self._scala_type_to_ink_type(scala_type.args[0])
                        value_type = self._scala_type_to_ink_type(scala_type.args[1])
                        return t.Map(key_type, value_type)
                case "List":
                    # List[T] -> List(T)
                    if len(scala_type.args) == 1:
                        elem_type = self._scala_type_to_ink_type(scala_type.args[0])
                        return t.List(elem_type)

                case "Set":
                    # Set[T] -> Set(T)
                    if len(scala_type.args) == 1:
                        elem_type = self._scala_type_to_ink_type(scala_type.args[0])
                        return t.Set(elem_type)

        match scala_type.name:
            case "String":
                return t.Str()
            case "Double" | "Float" | "Int" | "Long":
                return t.Num()
            case "Boolean":
                return t.Bool()
            case _:
                case_class = self._find_case_class_by_name(scala_type.name)
                if case_class:
                    return self._case_class_to_ink_type(case_class)
                else:
                    # For unknown types, just use Num as a fallback
                    # This allows the transpiler to continue even with unknown types
                    return t.Num()

    def _find_reduce_method(self, aggregator_class: s.ClassDecl) -> s.FunctionDecl:
        """Find the reduce method in the aggregator class."""
        for node in aggregator_class.body:
            if isinstance(node, s.FunctionDecl) and node.name == "reduce":
                return node
        raise ValueError("No reduce method found in Aggregator class")

    def _find_zero_method(self, aggregator_class: s.ClassDecl) -> s.FunctionDecl:
        """Find the zero method in the aggregator class."""
        for node in aggregator_class.body:
            if isinstance(node, s.FunctionDecl) and node.name == "zero":
                return node
        raise ValueError("No zero method found in Aggregator class")

    def _convert_reduce_method(self, reduce_method: s.FunctionDecl) -> ink.Expr:
        """Convert the reduce method body to an ink expression."""
        # Extract parameter names
        if len(reduce_method.params) != 2:
            raise ValueError("reduce method must have exactly 2 parameters")

        self.buffer_param_name = reduce_method.params[0].name
        self.input_param_name = reduce_method.params[1].name

        # Convert the method body
        return self._convert_statements(reduce_method.body)

    def _convert_zero_method(self, zero_method: s.FunctionDecl) -> ink.Expr:
        """Convert the zero method body to an ink expression."""
        # The zero method should have no parameters
        if zero_method.params is not None and len(zero_method.params) != 0:
            raise ValueError("zero method must have no parameters")

        # Convert the method body to get the initial value
        return self._convert_statements(zero_method.body)

    def _convert_statements(self, statements: list[s.ScalaNode]) -> ink.Expr:
        """Convert a sequence of statements to an ink expression."""
        # Collect let bindings and final expression
        let_bindings = []
        final_expr = None

        # Fix parsing issue with multi-line boolean expressions
        statements = self._fix_multiline_boolean_expressions(statements)

        for stmt in statements:
            if isinstance(stmt, s.VarDecl):
                # Variable declarations become let bindings
                let_bindings.append((stmt.name, self._convert_expression(stmt.value)))
            elif isinstance(stmt, s.Call):
                # Function call becomes the final expression
                final_expr = self._convert_expression(stmt)
            else:
                # Other expressions become the final expression
                final_expr = self._convert_expression(stmt)

        # Build the expression with let bindings
        if let_bindings:
            return ink.bind_lets(let_bindings, final_expr or ink.Var("unit"))
        else:
            return final_expr or ink.Var("unit")

    def _fix_multiline_boolean_expressions(
        self, statements: list[s.ScalaNode]
    ) -> list[s.ScalaNode]:
        """Fix parsing issues with multi-line boolean expressions that span multiple statements."""
        fixed_statements = []
        i = 0

        while i < len(statements):
            stmt = statements[i]

            # Check if this is a VarDecl with a Field value that has a logical operator
            if (
                isinstance(stmt, s.VarDecl)
                and isinstance(stmt.value, s.Field)
                and stmt.value.field in ("||", "&&")
            ):
                # This is the start of a broken multi-line boolean expression
                # Collect all the parts
                boolean_parts = []
                logical_op = stmt.value.field

                # Add the first part (from the VarDecl)
                boolean_parts.append(stmt.value.expr)

                # Look ahead and collect subsequent Field expressions with the same operator
                j = i + 1
                while (
                    j < len(statements)
                    and isinstance(statements[j], s.Field)
                    and statements[j].field == logical_op
                ):  # type: ignore
                    boolean_parts.append(statements[j].expr)  # type: ignore
                    j += 1

                # Handle the final part (might be an InfixExpr or another expression)
                if j < len(statements) and isinstance(statements[j], s.InfixExpr):
                    boolean_parts.append(statements[j])
                    j += 1

                # Reconstruct the boolean expression
                if len(boolean_parts) > 1:
                    # Build the expression from left to right
                    result_expr = boolean_parts[0]
                    for part in boolean_parts[1:]:
                        result_expr = s.InfixExpr(
                            node=stmt.node,  # Use the node from the original VarDecl
                            left=result_expr,
                            op=logical_op,
                            right=part,
                        )

                    # Create a new VarDecl with the fixed expression
                    fixed_stmt = s.VarDecl(
                        node=stmt.node,
                        name=stmt.name,
                        type=stmt.type,
                        value=result_expr,
                    )
                    fixed_statements.append(fixed_stmt)
                else:
                    # Fallback - just use the original statement
                    fixed_statements.append(stmt)

                # Skip the statements we've processed
                i = j
            else:
                # Regular statement, add as-is
                fixed_statements.append(stmt)
                i += 1

        return fixed_statements

    def _convert_expression(self, expr: s.ScalaNode) -> ink.Expr:
        """Convert a Scala expression to an ink expression."""
        match expr:
            case s.Var(name=name):
                # Special handling for Nil which represents empty list
                if name == "Nil":
                    return ink.Nil()
                return ink.Var(name)

            case s.Constant(value=value):
                if isinstance(value, bool):
                    return ink.Bool(value)
                elif isinstance(value, (int, float)):
                    return ink.Num(int(value))
                elif isinstance(value, str):
                    return ink.Str(value)
                else:
                    raise NotImplementedError(
                        f"Constant {value} type {type(value)} not supported"
                    )

            case s.Field(expr=obj, field=field_name):
                # Field access becomes tuple access, unless it's a unary tuple
                # Special handling for built-in methods that should be converted to function calls
                if field_name == "length":
                    obj_expr = self._convert_expression(obj)
                    return ink.App(ink.Var("length"), obj_expr)

                # Special handling for MaxValue and MinValue constants
                if isinstance(obj, s.Var) and field_name in (
                    "MaxValue",
                    "MinValue",
                    "MAX_VALUE",
                    "MIN_VALUE",
                ):
                    # Convert MAX_VALUE/MIN_VALUE to MaxValue/MinValue format
                    if field_name == "MAX_VALUE":
                        normalized_field = "MaxValue"
                    elif field_name == "MIN_VALUE":
                        normalized_field = "MinValue"
                    else:
                        normalized_field = field_name
                    return self._convert_max_min_value(obj.name, normalized_field)

                # Special handling for method calls on MaxValue/MinValue constants (e.g., Integer.MAX_VALUE.toLong)
                if (
                    isinstance(obj, s.Field)
                    and isinstance(obj.expr, s.Var)
                    and obj.field in ("MaxValue", "MinValue", "MAX_VALUE", "MIN_VALUE")
                    and field_name in ("toLong", "toDouble", "toInt", "toFloat")
                ):
                    # Convert the inner MAX_VALUE/MIN_VALUE and ignore the type conversion
                    if obj.field == "MAX_VALUE":
                        normalized_field = "MaxValue"
                    elif obj.field == "MIN_VALUE":
                        normalized_field = "MinValue"
                    else:
                        normalized_field = obj.field
                    return self._convert_max_min_value(obj.expr.name, normalized_field)

                # Special handling for empty collections (e.g., List.empty, Map.empty, Set.empty)
                if isinstance(obj, s.Var) and field_name == "empty":
                    if obj.name == "Map":
                        return ink.Var("empty_map")
                    elif obj.name == "Set":
                        return ink.Var("empty_set")
                    elif obj.name == "List":
                        return ink.Nil()
                    else:
                        # Fall through to normal field access
                        pass

                obj_expr = self._convert_expression(obj)

                # Check if this is a unary tuple (single field case class)
                if self._is_unary_tuple_field(obj, field_name):
                    return obj_expr  # For unary tuples, just return the object itself

                field_index = self._get_field_index(obj, field_name)
                return ink.TupleAccess(obj_expr, field_index)

            case s.InfixExpr(left=left, op=op, right=right):
                return self._convert_infix_expression(left, op, right)

            case s.PrefixExpr(op=op, expr=expr):
                return self._convert_prefix_expression(op, expr)

            case s.If(test=test, body=body, orelse=orelse):
                cond = self._convert_expression(test)
                then_expr = (
                    self._convert_statements(body)
                    if isinstance(body, list)
                    else self._convert_expression(body)
                )
                else_expr = (
                    self._convert_statements(orelse)
                    if isinstance(orelse, list)
                    else self._convert_expression(orelse)
                )
                return ink.Ite(cond, then_expr, else_expr)

            case s.Call(func=func, args=args):
                return self._convert_function_call(func, args)

            case s.Assignment(target=_, value=value):
                # For named parameters in constructor calls, we just use the value
                return self._convert_expression(value)

            case s.GenericFunction(expr=expr, types=types):
                return self._convert_generic_function(expr, types)

            case _:
                raise NotImplementedError(
                    f"Expression conversion not implemented for {type(expr).__name__}: {expr}"
                )

    def _convert_prefix_expression(self, op: str, expr: s.ScalaNode) -> ink.Expr:
        """Convert a prefix expression to ink."""
        expr_converted = self._convert_expression(expr)

        match op:
            case "!":
                return ink.UnaryOp(ink.UnaryOpKinds.NOT, expr_converted)
            case "-":
                return ink.UnaryOp(ink.UnaryOpKinds.NEG, expr_converted)
            case _:
                raise NotImplementedError(f"Prefix operator {op} not supported")

    def _convert_infix_expression(
        self, left: s.ScalaNode, op: str, right: s.ScalaNode
    ) -> ink.Expr:
        """Convert an infix expression to ink."""
        left_expr = self._convert_expression(left)
        right_expr = self._convert_expression(right)

        # Handle special cases that require AST node information
        match op:
            case "+":
                if self._is_set_type(left):
                    # Set + element becomes set_add(element, set)
                    return ink.call_params(ink.Var("set_add"), [right_expr, left_expr])
                # Check if this is a map operation
                elif self._is_map_operation(left):
                    # Map + (key -> value) becomes map[key <- value]
                    if (
                        isinstance(right_expr, ink.Tuple)
                        and len(right_expr.values) == 2
                    ):
                        key, value = right_expr.values
                        return ink.MapAssign(map=left_expr, key=key, value=value)
            case "++":
                # Handle ++ operator for different types
                # Check if this is a map concatenation
                if self._is_map_type(left) or self._is_map_type(right):
                    # Map ++ Map becomes concat_map(map1, map2)
                    return ink.call_params(
                        ink.Var("concat_map"), [left_expr, right_expr]
                    )

        return self._convert_infix_expression_with_exprs(left_expr, op, right_expr)

    def _convert_infix_expression_with_exprs(
        self, left_expr: ink.Expr, op: str, right_expr: ink.Expr
    ) -> ink.Expr:
        """Convert an infix expression to ink using already converted expressions."""
        match op:
            case "+":
                return ink.BinOp(ink.BinOpKinds.ADD, left_expr, right_expr)
            case "-":
                return ink.BinOp(ink.BinOpKinds.SUB, left_expr, right_expr)
            case "*":
                return ink.BinOp(ink.BinOpKinds.MUL, left_expr, right_expr)
            case "/":
                return ink.BinOp(ink.BinOpKinds.DIV, left_expr, right_expr)
            case "<":
                return ink.BinOp(ink.BinOpKinds.LT, left_expr, right_expr)
            case "<=":
                return ink.UnaryOp(
                    ink.UnaryOpKinds.NOT,
                    ink.BinOp(ink.BinOpKinds.GT, left_expr, right_expr),
                )
            case ">":
                return ink.BinOp(ink.BinOpKinds.GT, left_expr, right_expr)
            case ">=":
                return ink.UnaryOp(
                    ink.UnaryOpKinds.NOT,
                    ink.BinOp(ink.BinOpKinds.LT, left_expr, right_expr),
                )
            case "==":
                return ink.BinOp(ink.BinOpKinds.EQ, left_expr, right_expr)
            case "!=":
                return ink.UnaryOp(
                    ink.UnaryOpKinds.NOT,
                    ink.BinOp(ink.BinOpKinds.EQ, left_expr, right_expr),
                )
            case "&&":
                return ink.BinOp(ink.BinOpKinds.AND, left_expr, right_expr)
            case "||":
                return ink.BinOp(ink.BinOpKinds.OR, left_expr, right_expr)
            case "->":
                # Scala's arrow operator creates a key-value pair (tuple)
                return ink.Tuple((left_expr, right_expr))
            case ":+":
                # Scala's :+ operator appends an element to a list
                # list :+ element becomes list ++ [element]
                single_element_list = ink.Cons(right_expr, ink.Nil())
                return ink.BinOp(ink.BinOpKinds.CONCAT, left_expr, single_element_list)
            case "::":
                # Scala's :: operator prepends an element to a list
                # element :: list becomes element :: list
                return ink.Cons(left_expr, right_expr)
            case "++":
                return ink.BinOp(ink.BinOpKinds.CONCAT, left_expr, right_expr)
            case _:
                raise NotImplementedError(f"Infix operator {op} not supported")

    def _convert_function_call(self, func: Any, args: list[s.ScalaNode]) -> ink.Expr:
        """Convert a function call to ink."""
        if isinstance(func, str):
            # Constructor call - create tuple
            # Check if this is a case class constructor
            if self._is_case_class_constructor(func):
                arg_exprs = [self._convert_expression(arg) for arg in args]

                # For single-parameter case classes, return the value directly (unary tuple)
                case_class = self._find_case_class_by_name(func)
                if case_class and len(case_class.params) == 1:
                    return arg_exprs[0] if arg_exprs else ink.Var("_0")
                else:
                    return ink.Tuple(tuple(arg_exprs))

            # Arbitrary function call - treat as uninterpreted function
            func_var = ink.Var(func)
            arg_exprs = [self._convert_expression(arg) for arg in args]
            return ink.call_params(func_var, arg_exprs)

        elif isinstance(func, s.Field):
            # Method call like map.getOrElse(key, default)
            return self._convert_method_call(func, args)

        # Handle other function calls
        raise NotImplementedError(
            f"Function call conversion not implemented for {func}"
        )

    def _is_case_class_constructor(self, func_name: str) -> bool:
        """Check if a function name is a case class constructor."""
        if self.input_case_class and func_name == self.input_case_class.name:
            return True
        if self.buffer_case_class and func_name == self.buffer_case_class.name:
            return True
        # Check if it's any case class in the AST
        case_class = self._find_case_class_by_name(func_name)
        return case_class is not None

    def _is_unary_tuple_field(self, obj: s.ScalaNode, field_name: str) -> bool:
        """Check if this is a field access on a unary tuple (single-field case class)."""
        if isinstance(obj, s.Var):
            if obj.name == self.input_param_name and self.input_case_class:
                return len(self.input_case_class.params) == 1
            elif obj.name == self.buffer_param_name and self.buffer_case_class:
                return len(self.buffer_case_class.params) == 1
        elif isinstance(obj, s.Field):
            # Handle nested field access - check if the result type is a unary tuple
            result_type = self._get_field_result_type(obj)
            if result_type:
                case_class = self._find_case_class_by_name(result_type)
                return case_class is not None and len(case_class.params) == 1
        return False

    def _get_field_result_type(self, field_obj: s.Field) -> str | None:
        """Get the result type name of a field access."""
        if isinstance(field_obj.expr, s.Var):
            # Simple field access like buffer.field or input.field
            var_name = field_obj.expr.name
            field_name = field_obj.field

            if var_name == self.input_param_name and self.input_case_class:
                # Find the field type in input case class
                for param in self.input_case_class.params:
                    if param.name == field_name:
                        return param.type.name
            elif var_name == self.buffer_param_name and self.buffer_case_class:
                # Find the field type in buffer case class
                for param in self.buffer_case_class.params:
                    if param.name == field_name:
                        return param.type.name
        elif isinstance(field_obj.expr, s.Field):
            # Nested field access - recursively determine the type
            parent_type = self._get_field_result_type(field_obj.expr)
            if parent_type:
                parent_case_class = self._find_case_class_by_name(parent_type)
                if parent_case_class:
                    for param in parent_case_class.params:
                        if param.name == field_obj.field:
                            return param.type.name
        return None

    def _get_field_index(self, obj: s.ScalaNode, field_name: str) -> int:
        """Get the tuple index for a field access."""
        if isinstance(obj, s.Var):
            if obj.name == self.input_param_name and self.input_case_class:
                field_map = self.input_field_map
            elif obj.name == self.buffer_param_name and self.buffer_case_class:
                field_map = self.buffer_field_map
            elif obj.name == "_":
                # Handle lambda parameter in filter operations
                # Extract numeric part from field name (e.g., "_2" -> 1)
                if field_name.startswith("_") and field_name[1:].isdigit():
                    return int(field_name[1:]) - 1  # Convert to 0-based index
                else:
                    raise ValueError(f"Invalid tuple field access: {field_name}")
            else:
                # For unknown object names, try to treat as uninterpreted function calls
                # Return 0 as a fallback index
                return 0

            return field_map.get(field_name, 0)

        elif isinstance(obj, s.Field):
            # Handle nested field access like buffer.data.start_time
            # Need to find the case class for the nested field
            nested_case_class = self._get_nested_case_class(obj)
            if nested_case_class:
                nested_field_map = {
                    param.name: i for i, param in enumerate(nested_case_class.params)
                }
                return nested_field_map.get(field_name, 0)
            else:
                # Fallback - assume it's the first field
                return 0

        else:
            # For any other object type, return 0 as fallback
            return 0

    def _get_nested_case_class(self, field_obj: s.Field) -> s.CaseClassDecl | None:
        """Get the case class for a nested field access."""
        # For field_obj like buffer.data, we need to find what type 'data' is
        if isinstance(field_obj.expr, s.Var):
            # Base case: buffer.data
            base_case_class = None
            if field_obj.expr.name == self.input_param_name and self.input_case_class:
                base_case_class = self.input_case_class
            elif (
                field_obj.expr.name == self.buffer_param_name and self.buffer_case_class
            ):
                base_case_class = self.buffer_case_class

            if base_case_class:
                # Find the field in the base case class
                for param in base_case_class.params:
                    if param.name == field_obj.field:
                        # Get the type name and find the corresponding case class
                        return self._find_case_class_by_name(param.type.name)

        elif isinstance(field_obj.expr, s.Field):
            # Recursive case: buffer.data.something
            return self._get_nested_case_class(field_obj.expr)

        return None

    def _convert_method_call(
        self, method_field: s.Field, args: list[s.ScalaNode]
    ) -> ink.Expr:
        """Convert a method call like map.getOrElse(key, default) to ink."""
        obj_expr = self._convert_expression(method_field.expr)
        method_name = method_field.field
        arg_exprs = [self._convert_expression(arg) for arg in args]

        match method_name:
            case "getOrElse":
                # map.getOrElse(key, default) -> ITE(contains_key(map, key), map[key], default)
                if len(arg_exprs) != 2:
                    raise ValueError("getOrElse expects 2 arguments")
                key, default = arg_exprs

                # Use a simple map access with default for now
                # In a more sophisticated implementation, we might use ITE with contains_key
                return ink.MapAccess(map=obj_expr, key=key)

            case "filter":
                # map.filter(_._2 != 0) -> filter_values(lambda, map)
                if len(args) != 1:
                    raise ValueError("filter expects 1 argument")

                # Check if this is filtering on values (pattern: _._2)
                filter_expr = args[0]
                if self._is_filter_values_pattern(filter_expr):
                    # Convert the filter expression to a lambda
                    lambda_expr = self._convert_underscore_to_lambda(filter_expr)
                    return ink.App(
                        ink.App(ink.Var("filter_values"), lambda_expr), obj_expr
                    )
                else:
                    # General filter case - not implemented yet
                    raise NotImplementedError("General filter not implemented yet")

            case "abs":
                if len(arg_exprs) != 1:
                    raise ValueError("abs expects 1 argument")
                return ink.App(ink.Var("abs"), arg_exprs[0])

            case _:
                # For unknown methods, treat as uninterpreted function calls
                if len(arg_exprs) == 0:
                    # Method with no arguments - treat as field access
                    return ink.Var(f"{method_name}")
                else:
                    # Method with arguments - create function application
                    func_call = ink.Var(method_name)
                    for arg in [obj_expr] + arg_exprs:
                        func_call = ink.App(func_call, arg)
                    return func_call

    def _is_filter_values_pattern(self, arg: s.ScalaNode) -> bool:
        """Check if this is a filter pattern that should use filter_values (e.g., _._2 != 0)."""
        # Check if the argument contains a reference to _._2 which indicates filtering map values
        if isinstance(arg, s.InfixExpr):
            # Check if either side of the comparison references _._2
            return self._contains_lambda_value_access(
                arg.left
            ) or self._contains_lambda_value_access(arg.right)
        return False

    def _contains_lambda_value_access(self, node: s.ScalaNode) -> bool:
        """Check if a node contains _._2 access pattern."""
        if isinstance(node, s.Field):
            return (
                isinstance(node.expr, s.Var)
                and node.expr.name == "_"
                and node.field == "_2"
            )
        return False

    def _is_map_type(self, node: s.ScalaNode) -> bool:
        """Check if a node represents a map type."""
        if isinstance(node, s.Var):
            # Check if this is one of our known map parameters
            if node.name == self.input_param_name:
                return isinstance(self.input_type, t.Map)
            elif node.name == self.buffer_param_name:
                return isinstance(self.buffer_type, t.Map)
        elif isinstance(node, s.Call):
            # Check if this is a method call that returns a map (like filter)
            if isinstance(node.func, s.Field) and node.func.field == "filter":
                return self._is_map_type(node.func.expr)
        return False

    def _is_set_type(self, node: s.ScalaNode) -> bool:
        """Check if a node represents a set type."""
        if isinstance(node, s.Var):
            # Check if this is one of our known set parameters
            if node.name == self.input_param_name:
                return isinstance(self.input_type, t.Set)
            elif node.name == self.buffer_param_name:
                return isinstance(self.buffer_type, t.Set)
        elif isinstance(node, s.Field):
            # Check if this is a field access on a set type
            field_type = self._get_field_type(node.expr, node.field)
            return isinstance(field_type, t.Set)
        return False

    def _is_map_operation(self, node: s.ScalaNode) -> bool:
        """Check if a node represents a map expression."""
        # Check if this is a field access on a map type
        if isinstance(node, s.Field):
            # Get the type of the field being accessed
            field_type = self._get_field_type(node.expr, node.field)
            return isinstance(field_type, t.Map)
        return False

    def _get_field_type(self, obj: s.ScalaNode, field_name: str) -> t.Type:
        """Get the type of a field."""
        if isinstance(obj, s.Var):
            if obj.name == self.input_param_name and self.input_case_class:
                case_class = self.input_case_class
            elif obj.name == self.buffer_param_name and self.buffer_case_class:
                case_class = self.buffer_case_class
            else:
                return t.Num()  # Default fallback

            # Find the field in the case class
            if case_class:
                for param in case_class.params:
                    if param.name == field_name:
                        return self._scala_type_to_ink_type(param.type)

        return t.Num()  # Default fallback

    def _convert_underscore_to_lambda(self, expr: s.ScalaNode) -> ink.Expr:
        """Convert an expression with _ to a lambda function for filter_values."""
        return ink.Lambda(
            param="value",
            param_type=t.Num(),
            body=self._replace_underscore_in_expr(expr, "value"),
        )

    def _replace_underscore_in_expr(
        self, expr: s.ScalaNode, param_name: str
    ) -> ink.Expr:
        """Replace _ references in an expression with a parameter name."""
        match expr:
            case s.InfixExpr(left=left, op=op, right=right):
                # Replace underscores in both left and right sides
                left_expr = self._replace_underscore_in_expr(left, param_name)
                right_expr = self._replace_underscore_in_expr(right, param_name)

                return self._convert_infix_expression_with_exprs(
                    left_expr, op, right_expr
                )

            case s.Field(expr=obj, field=field_name):
                if isinstance(obj, s.Var) and obj.name == "_":
                    # For filter_values, _._2 (the value part) becomes just the lambda parameter
                    if field_name == "_2":
                        # _._2 becomes just 'value' (the value itself, not a tuple access)
                        return ink.Var(param_name)
                    elif field_name == "_1":
                        # _._1 would be the key part, but filter_values only works on values
                        raise NotImplementedError(
                            "filter_values only works on values (_._2), not keys (_._1)"
                        )
                    else:
                        raise NotImplementedError(
                            f"Field {field_name} not supported in lambda"
                        )
                else:
                    return self._convert_expression(expr)

            case _:
                return self._convert_expression(expr)

    def _convert_max_min_value(self, type_name: str, value_type: str) -> ink.Expr:
        """Convert MaxValue/MinValue constants to ink variables."""
        if value_type == "MaxValue":
            match type_name:
                case "Double" | "Float":
                    return ink.Var("_mx")
                case "Int" | "Integer" | "Long":
                    return ink.Var("_mx")
                case _:
                    return ink.Var("_mx")
        elif value_type == "MinValue":
            match type_name:
                case "Double" | "Float":
                    return ink.Var("_mn")
                case "Int" | "Integer" | "Long":
                    return ink.Var("_mn")
                case _:
                    return ink.Var("_mn")
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _convert_generic_function(
        self, expr: s.ScalaNode, types: list[s.Type]
    ) -> ink.Expr:
        """Convert a generic function call like Map.empty, Set.empty, List.empty to ink variables."""
        if isinstance(expr, s.Field):
            obj_name = (
                expr.expr.name if isinstance(expr.expr, s.Var) else str(expr.expr)
            )
            field_name = expr.field

            # Handle empty collections
            if field_name == "empty":
                if obj_name == "Map":
                    return ink.Var("empty_map")
                elif obj_name == "Set":
                    return ink.Var("empty_set")
                elif obj_name == "List":
                    return ink.Nil()
                else:
                    raise NotImplementedError(
                        f"Empty collection not supported for {obj_name}"
                    )
            else:
                raise NotImplementedError(
                    f"Generic function {obj_name}.{field_name} not supported"
                )
        else:
            raise NotImplementedError(
                f"Generic function with non-field expression not supported: {expr}"
            )
