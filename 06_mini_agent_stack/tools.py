from __future__ import annotations

import ast
import operator as op
from typing import Any, Dict


# Safe mini calculator adapted for classroom use.
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> Dict[str, Any]:
    parsed = ast.parse(expression, mode="eval")
    value = _eval_node(parsed.body)
    return {"expression": expression, "result": value}


# Fake weather backend, intentionally deterministic for teaching.
_FAKE_WEATHER = {
    ("lisbon", "today"): {"temperature_c": 18, "condition": "sunny"},
    ("lisbon", "tomorrow"): {"temperature_c": 17, "condition": "partly cloudy"},
    ("porto", "today"): {"temperature_c": 15, "condition": "rain"},
    ("porto", "tomorrow"): {"temperature_c": 16, "condition": "cloudy"},
}


def get_weather(city: str, date: str) -> Dict[str, Any]:
    key = (city.strip().lower(), date.strip().lower())
    payload = _FAKE_WEATHER.get(key)
    if payload is None:
        return {
            "city": city,
            "date": date,
            "status": "not_found",
            "message": "No weather data available in the fake backend for this query.",
        }
    return {
        "city": city,
        "date": date,
        "status": "ok",
        **payload,
    }


TOOLS = {
    "get_weather": get_weather,
    "calculator": calculator,
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    func = TOOLS[tool_name]
    return func(**arguments)
