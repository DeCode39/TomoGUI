from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data_model import ViewerError

_ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arctan2": np.arctan2,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
}
_ALLOWED_CONSTANTS = {
    "pi": np.pi,
}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not)
_ALLOWED_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_ALLOWED_BOOLOPS = (ast.And, ast.Or)


@dataclass(frozen=True)
class CompiledExpression:
    expression: str
    tree: ast.Expression
    references: dict[str, set[str]] = field(default_factory=dict)


class _Validator(ast.NodeVisitor):
    def __init__(self, available_symbols: dict[str, set[str]]) -> None:
        self.available_symbols = available_symbols
        self.references = {name: set() for name in available_symbols}

    def generic_visit(self, node: ast.AST) -> Any:
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Attribute,
            ast.Compare,
            ast.BoolOp,
        )
        if not isinstance(node, allowed):
            raise ViewerError(f"Unsupported expression element: {node.__class__.__name__}")
        return super().generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ViewerError(f"Operator '{node.op.__class__.__name__}' is not allowed.")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ViewerError(f"Unary operator '{node.op.__class__.__name__}' is not allowed.")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if not isinstance(node.op, _ALLOWED_BOOLOPS):
            raise ViewerError(f"Boolean operator '{node.op.__class__.__name__}' is not allowed.")
        for value in node.values:
            self.visit(value)

    def visit_Compare(self, node: ast.Compare) -> Any:
        self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, _ALLOWED_CMPOPS):
                raise ViewerError(f"Comparison operator '{op.__class__.__name__}' is not allowed.")
            self.visit(comparator)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise ViewerError("Only approved math functions are allowed in expressions.")
        for arg in node.args:
            self.visit(arg)
        for kwarg in node.keywords:
            self.visit(kwarg.value)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in _ALLOWED_FUNCTIONS or node.id in _ALLOWED_CONSTANTS:
            return
        if node.id in self.available_symbols:
            raise ViewerError(f"Use field syntax like {node.id}.vel instead of bare '{node.id}'.")
        raise ViewerError(f"Unknown symbol '{node.id}'.")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if not isinstance(node.value, ast.Name) or node.value.id not in self.available_symbols:
            raise ViewerError("Field references must look like A.vel or B.xi.")
        volume_name = node.value.id
        field_name = node.attr
        if field_name not in self.available_symbols[volume_name]:
            raise ViewerError(f"Field '{field_name}' is not available in volume {volume_name}.")
        self.references[volume_name].add(field_name)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if not isinstance(node.value, (int, float)):
            raise ViewerError("Only numeric constants are allowed in expressions.")


def compile_expression(expr: str, available_symbols: dict[str, set[str]]) -> CompiledExpression:
    expr = expr.strip()
    if not expr:
        raise ViewerError("Expression is empty.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ViewerError(f"Invalid expression syntax: {exc.msg}") from exc
    validator = _Validator(available_symbols)
    validator.visit(tree)
    references = {name: refs for name, refs in validator.references.items() if refs}
    if not references:
        raise ViewerError("Expression must reference at least one field.")
    return CompiledExpression(expression=expr, tree=tree, references=references)


def evaluate_expression(compiled: CompiledExpression, context: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    result = _evaluate_node(compiled.tree.body, context)
    result = np.asarray(result)
    if result.ndim != 3:
        raise ViewerError(f"Expression result must be 3D, got shape {result.shape}.")
    if result.dtype == np.bool_:
        result = result.astype(float)
    if not np.issubdtype(result.dtype, np.number):
        raise ViewerError("Expression result is not numeric.")
    return result


def _evaluate_node(node: ast.AST, context: dict[str, dict[str, np.ndarray]]) -> Any:
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, context)
        right = _evaluate_node(node.right, context)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ViewerError(f"Unsupported operator '{node.op.__class__.__name__}'.")

    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, context)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.Not):
            return np.logical_not(value)
        raise ViewerError(f"Unsupported unary operator '{node.op.__class__.__name__}'.")

    if isinstance(node, ast.BoolOp):
        values = [_evaluate_node(value, context) for value in node.values]
        if isinstance(node.op, ast.And):
            result = np.asarray(values[0], dtype=bool)
            for value in values[1:]:
                result = np.logical_and(result, np.asarray(value, dtype=bool))
            return result
        if isinstance(node.op, ast.Or):
            result = np.asarray(values[0], dtype=bool)
            for value in values[1:]:
                result = np.logical_or(result, np.asarray(value, dtype=bool))
            return result
        raise ViewerError(f"Unsupported boolean operator '{node.op.__class__.__name__}'.")

    if isinstance(node, ast.Compare):
        left = _evaluate_node(node.left, context)
        result: Any = True
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate_node(comparator, context)
            if isinstance(op, ast.Lt):
                comparison = left < right
            elif isinstance(op, ast.LtE):
                comparison = left <= right
            elif isinstance(op, ast.Gt):
                comparison = left > right
            elif isinstance(op, ast.GtE):
                comparison = left >= right
            elif isinstance(op, ast.Eq):
                comparison = left == right
            elif isinstance(op, ast.NotEq):
                comparison = left != right
            else:
                raise ViewerError(f"Unsupported comparison operator '{op.__class__.__name__}'.")
            result = np.logical_and(result, comparison)
            left = right
        return result

    if isinstance(node, ast.Call):
        func = _ALLOWED_FUNCTIONS[node.func.id]
        args = [_evaluate_node(arg, context) for arg in node.args]
        kwargs = {kw.arg: _evaluate_node(kw.value, context) for kw in node.keywords}
        return func(*args, **kwargs)

    if isinstance(node, ast.Attribute):
        assert isinstance(node.value, ast.Name)
        return context[node.value.id][node.attr]

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise ViewerError(f"Unknown symbol '{node.id}'.")

    if isinstance(node, ast.Constant):
        return node.value

    raise ViewerError(f"Unsupported node '{node.__class__.__name__}'.")
