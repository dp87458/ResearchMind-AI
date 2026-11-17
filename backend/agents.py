"""LangGraph nodes: Enhance -> Hybrid Search -> Reader -> Writer -> Critic -> (loop) -> Gap."""
import json
import os
import re

from langchain_groq import ChatGroq

from .state import ResearchState
from . import tools


def _llm(temp=0.3):
    return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=temp)


def _json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


# 1. Planning / Query-expansion node
def enhance_node(state: ResearchState):
    prompt = (
        "You are a research planner. Expand the topic into 3 diverse, specific academic "
        "search queries (different angles: methods, applications, recent advances).\n"
        f"Topic: {state['topic']}\n"
        'Return ONLY JSON: {"queries": ["q1","q2","q3"]}'
    )
    try:
        qs = _json(_llm(0.5).invoke(prompt).content).get("queries", [])
    except Exception:
        qs = []
    qs = [state["topic"]] + [q for q in qs if q][:3]
    return {"queries": qs, "revisions": 0, "max_revisions": state.get("max_revisions", 2)}


# 2. Hybrid search node: local ChromaDB + Tavily live web
def search_node(state: ResearchState):
    rag = tools.retrieve(state["session_id"], state["queries"])
    web, seen = [], set()
    for q in state["queries"][:3]:
        for r in tools.tavily_search(q):
            if r["url"] not in seen:
                seen.add(r["url"])
                web.append(r)
    return {"rag_context": rag, "web_results": web}


# 3. Reader node: deep scrape top URLs
def reader_node(state: ResearchState):
    scraped = []
    for r in state.get("web_results", [])[:4]:
        text = tools.scrape_url(r["url"])
        if text:
            scraped.append({"url": r["url"], "title": r["title"], "text": text})
    return {"scraped": scraped}


# 4. Writer node
def writer_node(state: ResearchState):
    web = "\n\n".join(
        f"[WEB: {r['title']} | {r['url']}] {r['content'][:300]}" for r in state.get("web_results", [])[:5]
    )
    deep = "\n\n".join(f"[SCRAPED: {s['url']}] {s['text']}" for s in state.get("scraped", []))
    fb = state.get("feedback", "")
    prompt = f"""You are an expert academic research writer. Write a structured Markdown literature report on:
"{state['topic']}"

Use ONLY the sources below. Cite inline like [PDF: file, p.X] or [WEB: url]. Never invent citations.
Sections: # Title, ## Overview, ## Literature Summary, ## Methodology Comparison (include a table), ## Key Findings, ## Future Work, ## References.

=== LOCAL PDF CONTEXT ===
{(state.get('rag_context') or '(no PDFs uploaded)')[:4000]}

=== WEB SNIPPETS ===
{web or '(none)'}

=== DEEP-SCRAPED PAGES ===
{(deep or '(none)')[:3000]}

{('=== PREVIOUS DRAFT ===' + chr(10) + state.get('draft', '')[:5000] + chr(10) + '=== CRITIC FEEDBACK (fix every point) ===' + chr(10) + fb) if fb else ''}
"""
    draft = _llm(0.3).invoke(prompt).content
    return {"draft": draft, "revisions": state.get("revisions", 0) + 1}


# 5. Critic node: score 1-10 + JSON feedback
def critic_node(state: ResearchState):
    prompt = f"""You are a strict peer reviewer. Evaluate this report for factual grounding in cited sources,
completeness, structure and clarity. Score 1-10 (10 = publishable).
Return ONLY JSON: {{"score": <int>, "feedback": "<specific fixes>"}}

REPORT:
{state['draft'][:6000]}"""
    try:
        out = _json(_llm(0).invoke(prompt).content)
        score, fb = int(out.get("score", 5)), out.get("feedback", "")
    except Exception:
        score, fb = 7, "Improve grounding and structure."
    return {"score": score, "feedback": fb}


def route_after_critic(state: ResearchState) -> str:
    """Self-correction loop: rewrite if score < 8 and revision limit not reached."""
    if state["score"] < 8 and state["revisions"] < state["max_revisions"]:
        return "rewrite"
    return "finish"


# 6. Gap node: extract research gaps from final draft
def gap_node(state: ResearchState):
    prompt = (
        "From this literature report, list 3-5 concrete RESEARCH GAPS as short Markdown bullets.\n\n"
        + state["draft"][:6000]
    )
    gaps = _llm(0.3).invoke(prompt).content
    return {"gaps": gaps, "draft": state["draft"] + "\n\n## Identified Research Gaps\n" + gaps}
