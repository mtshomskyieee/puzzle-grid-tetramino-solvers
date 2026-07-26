"""Step 5 — agents solver: the LangGraph client wiring imports and exposes its pieces.

This is a smoke test — it doesn't call an LLM or the API server. It only checks that the
agent module wires up (LangGraph + the transformer scorer dependency) and exposes the
building blocks the graph is assembled from.
"""
import replay_using_agents as agents


def test_agent_building_blocks_exist():
    for name in ("build_graph", "GameAPI", "AgentPlayState", "ChosenPlacement"):
        assert hasattr(agents, name), f"missing {name}"
    assert callable(agents.build_graph)


def test_chosen_placement_validates_fields():
    cp = agents.ChosenPlacement(row=18, col=3, rotation=0)
    assert (cp.row, cp.col, cp.rotation) == (18, 3, 0)
