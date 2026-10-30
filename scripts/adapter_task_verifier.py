"""Bounded subprocess verifier for the local, small pure-function tasks.

Accepts only function definitions, disallows imports/dunder access, and exposes
selected pure builtins. The parent also imposes a wall-clock timeout. This is
an integration-test guard, not a sandbox for arbitrary hostile Python.
"""
from __future__ import annotations

import ast
import builtins
import json
import re
import sys

SAFE_BUILTINS = "abs all any bool dict enumerate filter float int isinstance len list map max min range reversed round set sorted str sum tuple zip".split()
SAFE_METHODS = set("add append clear copy count endswith extend get index isalnum isalpha isdigit items join keys lower pop remove replace reverse sort split startswith strip upper values".split())


def verify(payload: dict) -> dict:
    text = payload["text"]
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S | re.I)
    source = match.group(1) if match else text.strip()
    tree = ast.parse(source)
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        if not isinstance(statement, ast.FunctionDef) or statement.decorator_list:
            raise ValueError("Only undecorated function definitions are allowed")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef)):
            raise ValueError("Unsupported Python operation")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder names are disallowed")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_METHODS:
            raise ValueError("Unsupported attribute: " + node.attr)
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in SAFE_BUILTINS}}
    exec(compile(tree, "<candidate>", "exec"), namespace)
    function = namespace.get(payload["function"])
    if not callable(function):
        raise ValueError("Expected function was not defined")
    rows = []
    for case in payload["cases"]:
        actual = function(*case["args"])
        # JSON normalization permits equivalent tuple/list interval outputs.
        actual = json.loads(json.dumps(actual, allow_nan=False))
        passed = actual == case["expected"] and type(actual) is type(case["expected"])
        rows.append({"args": case["args"], "expected": case["expected"], "actual": actual, "passed": passed})
    return {"passed": all(row["passed"] for row in rows), "tests": rows,
            "reason": "All function tests passed" if all(row["passed"] for row in rows) else "Function tests failed"}


if __name__ == "__main__":
    try:
        result = verify(json.loads(sys.stdin.read()))
    except Exception as exc:
        result = {"passed": False, "tests": [], "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, allow_nan=False))
