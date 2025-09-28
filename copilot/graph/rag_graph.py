from typing import TypedDict, Literal, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from copilot.llm.provider import get_chat
from copilot.prompts.answer_prompt import ANSWER_SYSTEM_PROMPT, ANSWER_HUMAN_PROMPT
from copilot.prompts.grade_prompt import GRADE_SYSTEM_PROMPT, GRADE_HUMAN_PROMPT
from copilot.prompts.judge_prompt import JUDGE_SYSTEM_PROMPT, JUDGE_HUMAN_PROMPT
from copilot.prompts.rewrite_prompt import REWRITE_SYSTEM_PROMPT, REWRITE_HUMAN_PROMPT
from copilot.rag.retriever import get_retriever


# ----- State -----
class QAState(TypedDict, total=False):  # total=False để các trường không bắt buộc phải có, có thể bổ sung dần
    question: str
    rewritten: str
    context: List[Document]
    answer: str
    k: int  # số tài liệu lấy về
    iterations: int  # số vòng đã lặp
    verdict: Literal["good", "retry"]  # verdict: phán quyết


# ----- Node: retrieve -----
def retrieve_node(state: QAState) -> QAState:
    q = state.get("rewritten", None) or state["question"]
    k = state.get("k", 6)
    retriever = get_retriever(k)
    docs = retriever.invoke(q)
    return {"context": docs}


# ----- Node: grade (lọc tài liệu) -----
_GRADER = get_chat("GRADE", temperature=0)

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GRADE_SYSTEM_PROMPT),
    ("human", GRADE_HUMAN_PROMPT)
])


def grade_node(state: QAState) -> QAState:
    q = state.get("rewritten", None) or state["question"]
    docs = state["context"]
    kept: List[Document] = []
    grand_chain = GRADER_PROMPT | _GRADER | (lambda x: x.content.strip().upper())
    for d in docs:
        res = grand_chain.invoke({
            "question": q,
            "doc": d
        })
        if res.startswith("Y"):
            kept.append(d)

        if not kept:
            kept = docs[:3]

    return {"context": kept}


# ----- Node: answer -----
_ANSWER_LLM = get_chat("ANSWER", temperature=0.2)  # Tăng độ sáng tạo một chút

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    ("human", ANSWER_HUMAN_PROMPT)
])


def _format_context(docs: List[Document]) -> str:
    part = []
    for i, d in enumerate(docs):
        snippet = d.page_content.strip().replace("\n", " ")
        src = d.metadata.get("source", "")
        part.append(f"[{i+1}] {snippet} (SOURCE: {src})")

    return "\n\n".join(part) if part else "(Không có ngữ cảnh)"


def answer_node(state: QAState) -> QAState:
    q = state.get("rewritten", None) or state["question"]
    docs = state["context"]
    context = _format_context(docs)
    answer_chain = _ANSWER_PROMPT | _ANSWER_LLM | (lambda x: x.content.strip())
    res = answer_chain.invoke({
        "question": q,
        "context": context
    })
    return {"answer": res}


# ----- Node: judge -----
_JUDGE_LLM = get_chat("JUDGE", temperature=0)

_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM_PROMPT),
    ("human", JUDGE_HUMAN_PROMPT)
])


def judge_node(state: QAState) -> QAState:
    q = state.get("rewritten", None) or state["question"]
    docs = state["context"]
    context = _format_context(docs)
    ans = state["answer"]
    judge_chain = _JUDGE_PROMPT | _JUDGE_LLM | (lambda x: x.content.strip().upper())
    res = judge_chain.invoke({
        "question": q,
        "context": context,
        "answer": ans
    })
    verdict: Literal["good", "retry"] = "good" if res.startswith("G") else "retry"
    return {"verdict": verdict}


# ----- Node: rewrite_query (khi RETRY) -----
_REWRITER = get_chat("REWRITE", temperature=0)

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYSTEM_PROMPT),
    ("human", REWRITE_HUMAN_PROMPT)
])


def rewrite_node(state: QAState) -> QAState:
    q = state["question"]
    rewriter_chain = REWRITE_PROMPT | _REWRITER | (lambda x: x.content.strip())
    res = rewriter_chain.invoke({
        "question": q
    })
    iters = state.get("iterations", 0) + 1
    return {"rewritten": res, "iterations": iters}


# ----- Build graph -----
def build_graph(max_iters: int = 2):
    sg = StateGraph(QAState)

    sg.add_node("retrieve", retrieve_node)
    sg.add_node("grade", grade_node)
    sg.add_node("answer", answer_node)
    sg.add_node("judge", judge_node)
    sg.add_node("rewrite_query", rewrite_node)

    sg.add_edge(START, "retrieve")
    sg.add_edge("retrieve", "grade")
    sg.add_edge("grade", "answer")
    sg.add_edge("answer", "judge")

    def should_retry(state: QAState) -> Literal["end", "rewrite"]:
        if state.get("verdict") == "good":
            return "end"
        if state.get("iterations", 0) >= max_iters:
            return "end"
        return "rewrite"

    sg.add_conditional_edges(
        "judge",
        should_retry,
        {
            "end": END,  # Kết thúc
            "rewrite": "rewrite_query"
        }
    )
    sg.add_edge("rewrite_query", "retrieve")

    # Lưu memory để debug đơn giản
    memory = MemorySaver()
    app = sg.compile(checkpointer=memory)

    return app, memory
