import json

from django.http import HttpResponseBadRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from copilot.graph.run import run_graph
from copilot.rag.settings import TOP_K


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