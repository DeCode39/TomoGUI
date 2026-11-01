"""Explicit, scalar quality-to-weight rules (no Python eval or exec)."""
from __future__ import annotations

import ast
import math
import operator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow}
_FUNCTIONS = {"min": min, "max": max, "abs": abs,
              "clip": lambda value, lo, hi: max(lo, min(value, hi))}


def finite_weight(value) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Weights must be finite and nonnegative.")
    return value


def formula(source: str, quality: float) -> float:
    if len(source) > 512:
        raise ValueError("The weight formula is too long.")
    tree = ast.parse(source, mode="eval")
    if len(list(ast.walk(tree))) > 100:
        raise ValueError("The weight formula is too complex.")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "q":
            return quality
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
            return _FUNCTIONS[node.func.id](*(visit(arg) for arg in node.args))
        raise ValueError("Use q, numbers, arithmetic, min, max, abs, or clip in the formula.")

    try:
        return finite_weight(visit(tree.body))
    except (ArithmeticError, TypeError, SyntaxError) as exc:
        raise ValueError(f"Invalid weight formula: {exc}") from exc


def quality_weight(value, original: float, spec: dict | None) -> float:
    if not spec or not spec.get("column"):
        return finite_weight(original)
    missing = value is None or str(value).strip().lower() in {'', 'nan', 'none', 'null'}
    if missing:
        policy = spec.get("missing", "error")
        if policy == "keep":
            return finite_weight(original)
        if policy == "zero":
            return 0.0
        raise ValueError("Missing quality value: choose a missing-value rule or fix the input.")
    if spec.get('mode') == 'grades':
        grade = str(value).strip().casefold()
        for rule in spec.get('grades', []):
            if grade == str(rule.get('grade', '')).strip().casefold():
                return finite_weight(rule['weight'])
        return finite_weight(spec.get('fallback', 0.0))
    try:
        q = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Quality contains text. Choose Alphabetical grades to map labels to weights.') from exc
    if not math.isfinite(q):
        raise ValueError('Quality must be finite.')
    if spec.get("mode") == "formula":
        return formula(spec.get("formula", ""), q)
    comparisons = {"=": operator.eq, "==": operator.eq, "<=": operator.le,
                   ">=": operator.ge, "<": operator.lt, ">": operator.gt}
    for rule in spec.get("rules", []):
        op = rule.get("op", "=")
        if op not in comparisons:
            raise ValueError("Unsupported quality comparison.")
        if comparisons[op](q, float(rule["value"])):
            return finite_weight(rule["weight"])
    return finite_weight(spec.get("fallback", 0.0))
