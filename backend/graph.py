from langgraph.graph import StateGraph, END
from .state import ResearchState
from .agents import (enhance_node, search_node, reader_node, writer_node,
                     critic_node, gap_node, route_after_critic)


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("enhance", enhance_node)
    g.add_node("search", search_node)
    g.add_node("reader", reader_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.add_node("gap", gap_node)

    g.set_entry_point("enhance")
    g.add_edge("enhance", "search")
    g.add_edge("search", "reader")
    g.add_edge("reader", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"rewrite": "writer", "finish": "gap"})
    g.add_edge("gap", END)
    return g.compile()


graph = build_graph()
