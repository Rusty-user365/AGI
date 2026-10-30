"""Small, disclosed integration tasks; these are not a held-out benchmark."""
from __future__ import annotations

import copy


TASKS = {
    "clamp": {
        "id": "clamp", "title": "Clamp a number", "required_skill": "coder",
        "prompt": "Write only Python code for clamp(value, lower, upper). Return value limited to the inclusive range [lower, upper]. Assume lower <= upper. No imports or examples.",
        "function": "clamp",
        "cases": [
            {"args": [5, 0, 10], "expected": 5},
            {"args": [-3, 0, 10], "expected": 0},
            {"args": [20, 0, 10], "expected": 10},
            {"args": [2, 2, 2], "expected": 2},
            {"args": [-4, -3, -1], "expected": -3},
        ],
    },
    "even_sum": {
        "id": "even_sum", "title": "Sum even numbers", "required_skill": "coder",
        "prompt": "Write only Python code for sum_even(numbers). Return the sum of even integers in numbers, including negative even numbers. Return 0 for an empty list. No imports or examples.",
        "function": "sum_even",
        "cases": [
            {"args": [[]], "expected": 0},
            {"args": [[1, 2, 3, 4]], "expected": 6},
            {"args": [[-4, -3, -2, 0, 5]], "expected": -6},
            {"args": [[1, 3, 5]], "expected": 0},
        ],
    },
    "brackets": {
        "id": "brackets", "title": "Validate brackets", "required_skill": "coder",
        "prompt": "Write only Python code for valid_brackets(text). Input contains only ()[]{}. Return True if brackets are correctly matched and nested; otherwise False. Empty input is valid. No imports or examples.",
        "function": "valid_brackets",
        "cases": [
            {"args": [""], "expected": True},
            {"args": ["([]{})"], "expected": True},
            {"args": ["([)]"], "expected": False},
            {"args": [")("], "expected": False},
            {"args": ["(("], "expected": False},
        ],
    },
    "rollback_control": {
        "id": "rollback_control", "title": "Injected failure / rollback control",
        "required_skill": "coder", "function": "add",
        "prompt": "Write only Python code for add(a, b), returning the sum of a and b. No imports or examples.",
        "injected_candidate": "def add(a, b):\n    return a - b",
        "cases": [{"args": [2, 3], "expected": 5}, {"args": [-1, 1], "expected": 0}],
    },
}


def base_node(node_id: str, reply: str) -> dict:
    return {"id": node_id, "title": "Base conversation: " + reply,
            "required_skill": "base", "prompt": f"Reply with exactly {reply} and nothing else.",
            "expected_text": reply}


def task_sequence(selection: str) -> list[dict]:
    if selection == "switch_suite":
        nodes = [base_node("ready", "READY"), TASKS["clamp"],
                 base_node("ack", "ACK"), TASKS["even_sum"], base_node("done", "DONE")]
    elif selection in TASKS:
        nodes = [base_node("ready", "READY"), TASKS[selection], base_node("done", "DONE")]
    elif selection == "unknown_control":
        nodes = [{"id": "unknown", "title": "Unknown skill / abstention control",
                  "required_skill": "unknown", "prompt": "Unsupported task"}]
    else:
        raise ValueError("Unknown task selection")
    return copy.deepcopy(nodes)


def catalog() -> list[dict]:
    return [{"id": "switch_suite", "title": "Switching sequence: base / coder / base / coder / base"}] + [
        {"id": key, "title": value["title"]} for key, value in TASKS.items()
    ] + [{"id": "unknown_control", "title": "Unknown skill / abstention control"}]
