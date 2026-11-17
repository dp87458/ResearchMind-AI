"""Global state schema shared by every LangGraph node (ResearchState)."""
from typing import TypedDict, List, Dict


class ResearchState(TypedDict, total=False):
    session_id: str          # which uploaded-PDF vector store to use
    topic: str               # original user topic
    queries: List[str]       # expanded query variations
    rag_context: str         # chunks retrieved from ChromaDB (local PDFs)
    web_results: List[Dict]  # Tavily results: title, url, content
    scraped: List[Dict]      # BeautifulSoup deep-scraped pages: url, text
    draft: str               # current Markdown report
    score: int               # critic score 1-10
    feedback: str            # critic feedback for the writer
    revisions: int           # number of writer passes so far
    max_revisions: int       # revision limit
    gaps: str                # research gaps (gap node)
