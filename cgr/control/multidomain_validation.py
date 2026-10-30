"""Public integration fixtures and deterministic SQL/static-HTML contracts."""
from __future__ import annotations

import html
import sqlite3
from html.parser import HTMLParser

from .adapter_runtime import verify_candidate


def graph_nodes():
    parents = {1: [], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6],
               8: [2], 9: [7], 10: [9], 11: [10], 12: [8], 13: [11],
               14: [12, 13], 15: [14], 16: [2], 17: [15], 18: [17], 19: [18], 20: [19, 16]}
    return [{"id": str(n), "parents": list(map(str, parents[n])),
             "skill": {6: "coder", 10: "cyber_defensive", 14: "ui"}.get(n, "base")}
            for n in range(1, 21)]


def scenarios():
    rows = []
    for index in range(20):
        lower = [-10, 0, 10, 100][index % 4]
        upper = lower + [1, 5, 10, 100, 1000][index // 4]
        value = lower - (index + 1)
        name = ["Alice", "O'Reilly", '<img src=x onerror="alert(1)">', "किरण & team"][index % 4]
        rows.append({"id": f"card_{index:02d}", "family": "sanitized_numeric_card",
                     "purpose": "public_integration_fixture", "lower": lower, "upper": upper,
                     "value": value, "expected": lower, "display_name": name,
                     "coder_prompt": "Return only Python code defining clamp(value, lower, upper). Limit value to the inclusive range. No imports.",
                     "cyber_prompt": 'Return a JSON object with sql and params. Select id,name from users by the supplied name, using one SQLite parameter. Represent the runtime input with params: ["$input"].',
                     "ui_prompt": f'Return only an HTML article containing a label for input id="amount", a numeric input with min="{lower}", max="{upper}", value="{lower}", an output role="status" containing "{lower}", and a span data-field="name" displaying this exact untrusted name as text: {name}. No scripts, styles, or event handlers.'})
    return rows


def coder_verdict(row, source):
    lo, hi = row["lower"], row["upper"]
    node = {"function": "clamp", "cases": [
        {"args": [lo - 1, lo, hi], "expected": lo},
        {"args": [hi + 1, lo, hi], "expected": hi},
        {"args": [lo, lo, hi], "expected": lo},
        {"args": [hi, lo, hi], "expected": hi},
    ]}
    return verify_candidate(node, source)


def cyber_verdict(candidate):
    if not isinstance(candidate, dict) or candidate.get("params") != ["$input"] or not isinstance(candidate.get("sql"), str):
        return {"passed": False, "reason": "Expected sql plus params ['$input']"}
    connection = sqlite3.connect(":memory:")
    names = ["Alice", "O'Reilly", "' OR 1=1 --", "किरण"]
    try:
        connection.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        connection.executemany("INSERT INTO users VALUES (?, ?)", list(enumerate(names)))
        def authorizer(action, arg1, arg2, database, trigger):
            allowed = action == sqlite3.SQLITE_SELECT or (action == sqlite3.SQLITE_READ and arg1 == "users" and arg2 in ("id", "name"))
            return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY
        connection.set_authorizer(authorizer)
        remaining = [1000]
        def deadline():
            remaining[0] -= 1
            return int(remaining[0] <= 0)
        connection.set_progress_handler(deadline, 100)
        tests = []
        for name in names + ["missing"]:
            actual = connection.execute(candidate["sql"], (name,)).fetchmany(10)
            expected = [(i, item) for i, item in enumerate(names) if item == name]
            tests.append({"input": name, "passed": actual == expected})
        return {"passed": all(t["passed"] for t in tests), "tests": tests}
    except sqlite3.Error as exc:
        return {"passed": False, "reason": str(exc)}
    finally:
        connection.close()


class CardParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes, self.stack, self.errors = [], [], []

    def handle_starttag(self, tag, attrs):
        if tag not in {"article", "label", "input", "output", "span", "p", "h2", "div"}:
            self.errors.append("Disallowed element: " + tag)
        if len(dict(attrs)) != len(attrs):
            self.errors.append("Duplicate attribute")
        allowed = {"id", "for", "type", "min", "max", "value", "role", "class", "data-field", "aria-label"}
        if any(key not in allowed for key, value in attrs):
            self.errors.append("Unsupported attribute")
        node = {"tag": tag, "attrs": dict(attrs), "text": ""}
        self.nodes.append(node)
        if tag != "input":
            self.stack.append(node)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1]["tag"] != tag:
            self.errors.append("Unbalanced markup")
        else:
            self.stack.pop()

    def handle_data(self, data):
        for node in self.stack:
            node["text"] += data


def ui_verdict(row, source):
    parser = CardParser()
    try:
        parser.feed(source)
        parser.close()
        nodes = parser.nodes
        inputs = [n for n in nodes if n["tag"] == "input" and n["attrs"].get("id") == "amount"]
        labels = [n for n in nodes if n["tag"] == "label" and n["attrs"].get("for") == "amount" and n["text"].strip()]
        names = [n for n in nodes if n["attrs"].get("data-field") == "name"]
        statuses = [n for n in nodes if n["tag"] == "output" and n["attrs"].get("role") == "status"]
        ids = [n["attrs"]["id"] for n in nodes if "id" in n["attrs"]]
        checks = {"allowed_markup": not parser.errors and not parser.stack,
                  "unique_ids": len(ids) == len(set(ids)),
                  "article": len([n for n in nodes if n["tag"] == "article"]) == 1,
                  "associated_label": len(labels) == 1 and len(inputs) == 1,
                  "safe_name_text": len(names) == 1 and names[0]["text"] == row["display_name"],
                  "status_value": len(statuses) == 1 and statuses[0]["text"].strip() == str(row["expected"])}
        checks["numeric_contract"] = len(inputs) == 1 and all(inputs[0]["attrs"].get(k) == str(v) for k, v in
            {"type": "number", "min": row["lower"], "max": row["upper"], "value": row["expected"]}.items())
        return {"passed": all(checks.values()), "checks": checks, "errors": parser.errors}
    except (TypeError, ValueError) as exc:
        return {"passed": False, "reason": str(exc)}


def golden_card(row):
    return (f'<article><label for="amount">Amount</label><input id="amount" type="number" '
            f'min="{row["lower"]}" max="{row["upper"]}" value="{row["expected"]}">'
            f'<output role="status">{row["expected"]}</output>'
            f'<span data-field="name">{html.escape(row["display_name"])}</span></article>')
