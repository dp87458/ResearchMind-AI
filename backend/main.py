"""FastAPI app: /api/upload (PDFs -> ChromaDB) and /api/research (SSE stream of agent steps)."""
import json
import os
import shutil
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from .graph import graph          # noqa: E402  (after load_dotenv)
from . import tools               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="ResearchMind AI")


class ResearchReq(BaseModel):
    session_id: str = ""
    topic: str
    max_revisions: int = 2


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    sid = uuid.uuid4().hex[:10]
    folder = os.path.join(UPLOAD_DIR, sid)
    os.makedirs(folder, exist_ok=True)
    paths = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        p = os.path.join(folder, os.path.basename(f.filename))
        with open(p, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(p)
    if not paths:
        raise HTTPException(400, "Upload at least one PDF")
    n = tools.build_vectorstore(sid, paths)
    return {"session_id": sid, "files": len(paths), "chunks": n}


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/research")
def research(req: ResearchReq):
    init = {
        "session_id": req.session_id or "none",
        "topic": req.topic,
        "max_revisions": req.max_revisions,
    }

    def gen():
        final = dict(init)
        try:
            for event in graph.stream(init, stream_mode="updates"):
                for node, upd in event.items():
                    final.update(upd)
                    info = {"node": node}
                    if node == "enhance":
                        info["detail"] = upd.get("queries")
                    elif node == "search":
                        info["detail"] = (f"{len(upd.get('web_results', []))} web results, "
                                          f"{'PDF context found' if upd.get('rag_context') else 'no PDF context'}")
                    elif node == "reader":
                        info["detail"] = f"{len(upd.get('scraped', []))} pages scraped"
                    elif node == "writer":
                        info["detail"] = f"draft #{upd.get('revisions')}"
                    elif node == "critic":
                        info["detail"] = {"score": upd.get("score"), "feedback": upd.get("feedback")}
                    elif node == "gap":
                        info["detail"] = "gaps extracted"
                    yield _sse({"type": "step", **info})
            sources = [{"title": r["title"], "url": r["url"]} for r in final.get("web_results", [])]
            yield _sse({"type": "done", "report": final.get("draft", ""),
                        "score": final.get("score"), "revisions": final.get("revisions"),
                        "sources": sources})
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "frontend")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "frontend", "index.html"))
