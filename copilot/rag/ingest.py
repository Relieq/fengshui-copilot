import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader

from copilot.llm.embeddings import get_embeddings
from copilot.rag.settings import CHUNK_SIZE, CHUNK_OVERLAP, ensure_dirs, CHROMA_DIR, EMBEDDING_MODEL, COLLECTION_NAME, \
    CORPUS_DIR
from copilot.rag.supa import *


_CONTROL_BAD = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')  # giữ \t \n \r


# Vệ sinh, loại bỏ control char không mong muốn
def sanitize_text(s: str) -> str:
    if not s:
        return ""
    # chuẩn hoá xuống 1 khoảng trắng với control char; strip cho gọn
    s = _CONTROL_BAD.sub(" ", s)
    return s.strip()


def _load_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf_file(path: Path) -> str:
    try:
        docs = PyMuPDFLoader(str(path)).load()
        return "\n".join(d.page_content for d in docs)
    except Exception:
        # Fallback: PyPDFLoader nếu PyMuPDF lỗi
        docs = PyPDFLoader(str(path)).load()
        return "\n".join(d.page_content for d in docs)


def load_corpus(corpus_dir: Path) -> List[Document]:
    EXTS = (".txt", ".md", ".markdown", ".mdx", ".pdf")
    docs = []

    for p in corpus_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith("."):
            try:
                if p.suffix.lower() == ".pdf":
                    content = _load_pdf_file(p)
                else:
                    content = _load_text_file(p)
            except Exception as e:
                print(f"[INGEST] Skip {p} ({e})")
                continue

            content = (content or "").strip() # phải (content or "") vì hàm _load_text_file có thể trả về None
            if not content:
                continue
            docs.append(Document(
                page_content=content,
                metadata={"source": str(p.relative_to(corpus_dir))} # đường dẫn tương đối đối với corpus_dir
                # để sau này hiện trích dẫn trong câu trả lời
            ))

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # <-- quan trọng: ghi vị trí bắt đầu chunk, lúc trước không thêm phần này bị lỗi trùng ID
    )
    return splitter.split_documents(docs)


def _make_uid(doc: Document) -> str:
    src = doc.metadata.get("source", "")
    start = doc.metadata.get("start_index", 0)
    return f"{src}::{start}"


def ingest_to_supabase(chunks: List[Document]) -> Tuple[int, int]:
    """
    Idempotent ingest:
    - Với mỗi source: lấy danh sách uid đang có trong DB.
    - Tạo uid hiện tại từ chunks.
      * new = current_uids - db_uids  -> chỉ embed + upsert cho phần này.
      * stale = db_uids - current_uids -> delete để làm sạch.
    - Không kiểm tra nội dung thay đổi (không checksum).
    """
    embeds = get_embeddings()
    client = get_supabase_client()
    table = get_supabase_table_name()

    # Kiểm tra metadata "source", "start_index"
    # for doc in chunks:
    #     print(f"[INGEST] {doc.metadata.get('source', '')} (start={doc.metadata.get('start_index', 0)})")

    by_src: Dict[str, List[Document]] = defaultdict(list)
    for doc in chunks:
        by_src[doc.metadata.get("source", "")].append(doc)

    # Kiểm tra các nguồn
    print(by_src.keys())

    total_new, total_delete = 0, 0

    for src, docs in by_src.items():
        res = client.table(table).select("uid").contains("metadata", {"source": src}).execute()
        db_uids = set([row["uid"] for row in (res.data or [])])

        cur_pairs = [(_make_uid(d), d) for d in docs]
        current_uids = set([uid for uid, _ in cur_pairs])

        # Xoá “stale” (những uid đang có trong DB nhưng không còn xuất hiện ở lần ingest này)
        stale = list(db_uids - current_uids)
        if stale:
            client.table(table).delete().in_("uid", stale).execute()
            total_delete += len(stale)

        # Chỉ embed + upsert những cái mới
        new_pairs = [(uid, d) for uid, d in cur_pairs if uid not in db_uids]
        if not new_pairs:
            continue

        content = [d.page_content for _, d in new_pairs]
        vectors = embeds.embed_documents(content)

        rows = []
        for (uid, d), vec in zip(new_pairs, vectors):
            rows.append({
                "uid": uid,
                "content": sanitize_text(d.page_content),
                "metadata": d.metadata,
                "embedding": vec
            })

        # Upsert theo batch để tránh payload quá lớn
        BATCH_SIZE = 128
        for i in range(0, len(rows), BATCH_SIZE):
            client.table(table).upsert(
                rows[i:i + BATCH_SIZE],
                on_conflict="uid"
            ).execute()

        total_new += len(new_pairs)

    return total_new, total_delete


    # ensure_dirs()
    #
    # if reset and CHROMA_DIR.exists():
    #     shutil.rmtree(CHROMA_DIR)

    # embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    #
    # # Chroma vector store
    # vs = Chroma(
    #     collection_name=COLLECTION_NAME,
    #     embedding_function=embeddings,
    #     persist_directory=str(CHROMA_DIR)
    # )
    #
    # # Tạo id ổn định để tránh trùng lặp nếu ingest nhiều lần
    # seen = set()
    # docs_unique, ids_unique = [], []
    # for doc in chunks:
    #     _id = _make_uid(doc)
    #     if _id in seen:
    #         continue
    #     seen.add(_id)
    #     docs_unique.append(doc)
    #     ids_unique.append(_id)
    #
    # # Chroma "add" sẽ bỏ qua id đã tồn tại (từ v0.5), an toàn khi ingest lại
    # vs.add_documents(docs_unique, ids=ids_unique)
    #
    # return len(ids_unique)


def ingest_corpus() -> dict:
    ensure_dirs()
    docs = load_corpus(CORPUS_DIR)
    chunks = chunk_documents(docs)
    # n = build_or_update_supabase(chunks)
    total_new, total_delete = ingest_to_supabase(chunks)

    return {
        "files": len(docs),
        "chunks": len(chunks),
        "added": total_new,
        "deleted": total_delete,
        "corpus_dir": str(CORPUS_DIR),
        # "chroma_dir": str(CHROMA_DIR),
        # "collection": COLLECTION_NAME,
    }
