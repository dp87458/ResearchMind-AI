"""Tools: PDF -> ChromaDB vector store, Tavily web search, BeautifulSoup scraping."""
import os
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# session_id -> in-memory (ephemeral) Chroma store
_STORES: Dict[str, Chroma] = {}


def build_vectorstore(session_id: str, pdf_paths: List[str]) -> int:
    """Load PDFs, chunk, embed (free local bge-small embeddings) and store in ephemeral ChromaDB."""
    docs = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(p).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0
    store = Chroma(
        collection_name=f"s_{session_id}",
        embedding_function=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    )  # no persist_directory => in-memory / ephemeral
    store.add_documents(chunks)
    _STORES[session_id] = store
    return len(chunks)


def retrieve(session_id: str, queries: List[str], k: int = 2) -> str:
    """k-NN similarity search over local PDFs for every query variation."""
    store = _STORES.get(session_id)
    if not store:
        return ""
    seen, out = set(), []
    for q in queries:
        for d in store.similarity_search(q, k=k):
            key = d.page_content[:80]
            if key in seen:
                continue
            seen.add(key)
            src = os.path.basename(d.metadata.get("source", "pdf"))
            page = d.metadata.get("page", "?")
            out.append(f"[PDF: {src}, p.{page}] {d.page_content}")
    return "\n\n".join(out)


def tavily_search(query: str, max_results: int = 4) -> List[Dict]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    try:
        from tavily import TavilyClient
        res = TavilyClient(api_key=key).search(query=query, max_results=max_results)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in res.get("results", [])
        ]
    except Exception as e:  # network / quota errors should not kill the run
        print("Tavily error:", e)
        return []


def scrape_url(url: str, limit: int = 1200) -> str:
    """Deep-scrape a page with BeautifulSoup for text beyond the search snippet."""
    if url.lower().endswith(".pdf"):
        return ""
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0 ResearchMindAI"})
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
            t.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:limit]
    except Exception as e:
        print("Scrape error:", url, e)
        return ""
