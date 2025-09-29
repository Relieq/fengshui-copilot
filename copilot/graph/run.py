import hashlib
from typing import List, Dict, Any
from uuid import uuid4

from langchain_core.documents import Document

from copilot.graph.rag_graph import build_graph
from copilot.rag.settings import TOP_K


def _uniq_sources(docs: List[Document], limit: int = 8):
    seen, out = set(), []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src in seen:
            continue
        seen.add(src)
        snippet = (d.page_content or "").strip().replace("\n", " ")
        out.append({"source": src, "snippet": snippet[:200]})
        if len(out) >= limit:
            break
    return out

def run_graph(question: str, k: int = TOP_K, max_iters: int = 2, tid: str | None = None,
              make_thread_id_from_question: bool = False) -> Dict[str, Any]:
    if make_thread_id_from_question:
        tid = "cli-" + hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    tid = tid or f"web-{uuid4()}"

    app, memory = build_graph(max_iters)

    state = {
        "question": question,
        "k": k,
        "iterations": 0
    }

    cfg = {
        "configurable":
            {
                "thread_id": tid,
            }
    }

    final = app.invoke(state, config=cfg)
    print("[INVOKE] DONE")
    print(f"Memory:")
    checkpoints = list(memory.list(cfg))
    for cp in checkpoints:
        print(cp)
    print()

    answer = final.get("answer", "").strip()
    verdict = final.get("verdict", "good")
    docs: List[Document] = final.get("context", [])
    sources = _uniq_sources(docs)

    return {"thread_id": tid, "answer": answer, "verdict": verdict, "sources": sources}
