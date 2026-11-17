# ResearchMind AI
**A Hierarchical Multi-Agent Retrieval-Augmented Generation Platform**
B.Tech Major Project · MSIT New Delhi · Dept. of EEE · Deepesh Rawat

ResearchMind AI bridges local PDF literature (ephemeral ChromaDB vectors) with live web intelligence
(Tavily + BeautifulSoup) using a LangGraph state machine with a Writer ⇄ Critic self-correction loop.

## Architecture
```
Enhance Topic → Hybrid Search (ChromaDB + Tavily) → Reader (scrape) → Writer → Critic
                                                           ▲              │ score < 8
                                                           └──────────────┘ (max 2 revisions)
                                                                          │ score ≥ 8
                                                                       Gap Node → END
```
| Layer | Tech |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (Llama 3.3 70B, free tier) |
| Vector DB | ChromaDB (in-memory) |
| Embeddings | Free local embeddings (FastEmbed `bge-small-en-v1.5`) |
| Web | Tavily API + BeautifulSoup |
| API / UI | FastAPI (SSE streaming) + HTML/CSS/JS |

## Run
```bash
python -m venv venv && venv\Scripts\activate      # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env                              # then add GROQ_API_KEY and TAVILY_API_KEY
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000

## Structure
```
backend/  state.py · tools.py · agents.py · graph.py · main.py
frontend/ index.html
```
## Roadmap
Citation-verification node · Ragas benchmarking (faithfulness, context precision/recall) vs single-stage RAG.
