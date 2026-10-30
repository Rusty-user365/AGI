import copy
import pytest
from cgr.control.dependency_graph import DependencyGraph
from cgr.control.multidomain_validation import graph_nodes, scenarios, cyber_verdict, ui_verdict, golden_card


def test_graph_rejects_cycles_and_missing_parents():
    for nodes in ([{"id": "a", "parents": ["b"], "skill": "base"}],
                  [{"id": "a", "parents": ["a"], "skill": "base"}]):
        with pytest.raises(ValueError):
            DependencyGraph(nodes)


def test_repair_invalidates_descendants_but_replays_independent_suffix():
    graph = DependencyGraph(graph_nodes())
    for n in graph.order:
        graph.commit(n, "v1:" + n, lambda _: True)
    before = copy.deepcopy(graph.artifacts)
    assert not graph.commit("6", "bad", lambda _: False)["accepted"]
    assert graph.artifacts == before and not graph.dirty
    result = graph.commit("6", "repaired", lambda _: True)
    assert "20" in result["invalidated"] and "8" not in result["invalidated"]
    with pytest.raises(ValueError):
        graph.commit("20", "stale", lambda _: True)
    plan = graph.cache_replay_plan("6", list(map(str, range(1, 21))), {str(n): n * 10 for n in range(1, 21)})
    assert plan["replay_preserved_artifacts"] == ["8", "12", "16"]
    assert "8" in plan["discard_kv_nodes"]


def test_attribution_retains_ambiguity_and_ignores_unrelated_failures():
    graph = DependencyGraph(graph_nodes())
    assert graph.fault_candidates("20", ["6", "7", "14", "20"]) == ["6"]
    assert set(graph.fault_candidates("20", ["6", "8"])) == {"6", "8"}
    assert graph.fault_candidates("7", ["8"]) == []


def test_sql_parameterization_requires_correct_behavior():
    good = {"sql": "SELECT id,name FROM users WHERE name = ?", "params": ["$input"]}
    assert cyber_verdict(good)["passed"]
    assert not cyber_verdict({**good, "sql": good["sql"] + " OR 1=1"})["passed"]
    assert not cyber_verdict({**good, "sql": "DELETE FROM users WHERE name = ?"})["passed"]
    assert not cyber_verdict({**good, "params": []})["passed"]


def test_ui_verifier_rejects_unescaped_names_wrong_labels_and_wrong_values():
    row = scenarios()[2]
    good = golden_card(row)
    assert ui_verdict(row, good)["passed"]
    for broken in (good.replace('for="amount"', 'for="other"'),
                   good.replace('type="number"', 'type="text"'),
                   good + '<script>alert(1)</script>',
                   good.replace('&lt;img src=x onerror=&quot;alert(1)&quot;&gt;', row["display_name"])):
        assert not ui_verdict(row, broken)["passed"]
