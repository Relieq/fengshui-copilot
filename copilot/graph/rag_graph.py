from typing import TypedDict, Literal, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from copilot.llm.provider import get_chat
from copilot.rag.retriever import get_retriever


# ----- State -----
class QAState(TypedDict, total=False): # total=False để các trường không bắt buộc phải có, có thể bổ sung dần
    question: str
    rewritten: str
    context: str
    answer: str
    k: int # số tài liệu lấy về
    iterations: int # số vòng đã lặp
    verdict: Literal["good", "retry"] # verdict: phán quyết

# ----- Node: retrieve -----
def retrieve_node(state: QAState) -> QAState:
    q = state["rewritten"] or state["question"]
    k = state.get("k", 6)
    retriever = get_retriever(k)
    docs = retriever.invoke(q)
    return {"context": docs}

# ----- Node: grade (lọc tài liệu) -----
_GRADER = get_chat(temperature=0)

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là bộ lọc tài liệu. Trả lời duy nhất YES hoặc NO."),
    ("human",
     "Câu hỏi: {question}\n\n"
     "Đoạn:\n{doc}\n\n"
     "Đoạn trên có hữu ích để trả lời câu hỏi không? Trả lời 'YES' hoặc 'NO'.")
])

def grade_node(state: QAState) -> QAState:
    q = state["rewritten"] or state["question"]
    docs = state["context"]
    kept: List[Document] = []
    grand_chain = GRADER_PROMPT | _GRADER | (lambda x: x.strip().upper())

# ----- Node: answer -----


# ----- Node: judge -----


# ----- Node: rewrite_query (khi RETRY) -----


# ----- Build graph -----
