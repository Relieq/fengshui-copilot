import json
import os
import queue
import threading
from typing import List

from django.http import HttpResponseBadRequest, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from copilot.graph.rag_graph import build_graph
from copilot.graph.run import run_graph
from copilot.llm.provider import get_chat
from copilot.prompts.answer_prompt import ANSWER_SYSTEM_PROMPT, ANSWER_HUMAN_PROMPT
from copilot.prompts.grade_prompt import GRADE_SYSTEM_PROMPT, GRADE_HUMAN_PROMPT
from copilot.rag.retriever import get_retriever
from copilot.rag.settings import TOP_K


def _sse(data: str, event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data:{data}\n\n"  # Nếu muốn chỉnh sửa ở đây thì xem xét tương ứng bên ask.html nha


@csrf_exempt
def api_ask_stream(req):
    if req.method != "POST":
        return HttpResponseBadRequest("POST only")

    try:
        payload = json.loads(req.body.decode('utf-8'))
        q = str(payload['question']).strip()
    except RuntimeError:
        return HttpResponseBadRequest("Missing 'question'")

    k = int(payload.get('k', TOP_K))
    max_iters = int(payload.get('max_iters', 2))
    thread_id = payload.get('thread_id', None)

    def event_stream():
        q_out: queue.Queue[tuple[str, str]] = queue.Queue()  # Thread-safe queue để giao tiếp giữa thread

        def emit(event: str, data):
            q_out.put((event, json.dumps(data) if not isinstance(data, str) else data))

        def run_graph_thread():
            # Không thể tái sử dụng run_graph vì nó không hỗ trợ streaming (không có "emit" callback)
            try:
                app, memory = build_graph(max_iters)
                tid = thread_id or f"sse-{os.getpid()}"
                print(f"[INVOKE] START thread_id={tid}")

                state = {
                    "question": q,
                    "k": k,
                    "iterations": 0,
                    "emit": emit
                    # Việc thêm Callable vào state sẽ gây lỗi nếu không điều chỉnh do Serializer của MemorySaver() do
                    # không thể xử lí loại đối tượng này
                }

                cfg = {
                    "configurable":
                        {
                            "thread_id": tid,
                        }
                }

                final = app.invoke(state, config=cfg)
                print(f"[INVOKE] Thread {tid} DONE")

                answer = final.get("answer", "").strip()
                verdict = final.get("verdict", "good")

                rest = {
                    "thread_id": tid,
                    "answer": answer,
                    "verdict": verdict,
                }

                q_out.put(("final", json.dumps(rest)))
            except Exception as e:
                # đẩy lỗi ra client để bạn thấy ngay trong UI
                q_out.put(("error", json.dumps({"type": type(e).__name__, "msg": str(e)}, ensure_ascii=False)))

            q_out.put(("__done__", ""))

        # Tạo thread chạy song song, chú ý phải đặt daemon=True để nó nếu ngắt chương trình thì thread này cũng dừng
        # theo, nếu không chương trình chỉ thoát sau khi tất cả non-daemon threads kết thúc (hoặc bị join()).
        t = threading.Thread(target=run_graph_thread, daemon=True)
        t.start()

        while True:
            ev, data = q_out.get()
            if ev == "__done__":
                break
            yield _sse(data, ev)

    # Trả response SSE để client (như brower) nhận từng event
    resp = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    # giúp proxy/nginx không buffer SSE
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


@csrf_exempt
def api_ask(req):
    if req.method != "POST":
        return HttpResponseBadRequest("POST only")

    try:
        data = json.loads(req.body.decode('utf-8'))
        q = str(data['question']).strip()
    except RuntimeError:
        return HttpResponseBadRequest("Missing 'question'")

    k = int(data.get('k', TOP_K))
    max_iters = int(data.get('max_iters', 2))
    thread_id = data.get('thread_id', None)

    try:
        result = run_graph(q, k, max_iters, tid=thread_id, make_thread_id_from_question=True)
        return JsonResponse({"ok": True, **result})
    except Exception as e:
        return JsonResponse({"ok": False, "error": type(e).__name__, "detail": str(e)}, status=500)
