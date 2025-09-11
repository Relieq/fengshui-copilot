Project này được tạo ra chủ yếu để học về LangChain và LangGraph. 

**Features**
* Q&A phong thủy có trích dẫn nguồn (RAG)
* Tool: đổi lịch, gợi ý ngũ hành/màu...
* LangGraph: tự chấm điểm, re-retrieve khi cần.

Project này sẽ được thực hiện thông qua chuỗi bài học sau:
* Bài 1 — Chuẩn bị môi trường + Django skeleton + Hello LLM
* Bài 2 — LangChain căn bản (Prompt → Model → Output Parser)
* Bài 3 — RAG v1: ingest tài liệu phong thủy (local)
* Bài 4 — Đánh giá nhỏ
* Bài 5 — LangGraph: vòng lặp tự-chấm
* Bài 6 — Tool Use + Router
* Bài 7 — Django UI
* Bài 8 — Theo dõi & cấu hình
* Bài 9 — Đóng gói & bàn giao

_Chú ý:_ Cài đặt các package cần thiết được liệt kê trong requirements.txt

# Bài 1: Chuẩn bị môi trường + Django skeleton + Hello LLM
## Mục tiêu
* Tạo venv, cài base: Django, langchain, langgraph, chromadb, python-dotenv
* Tạo app copilot
* Tạo lệnh quản trị hello_llm gọi LLM (Ollama): in câu chào, log thời gian phản hồi.

## Các bước thực hiện
1. Cài Ollama, pull mô hình về (các bạn có thể lựa chọn mô hình khác):
```bash
ollama pull llama3.1:8b
```
2. Cài đặt package, tạo app copilot: python manage.py startapp copilot
3. Nạp .env trước khi Django chạy
```python
def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fengshui.settings")
    load_dotenv() # nạp .env
    try:...
```

4. Tạo lệnh quản trị hello_llm trong copilot/management/commands/hello_llm.py
**Django management command là gì?**
* Là lệnh CLI tùy biến chạy qua `python manage.py <tên_lệnh>`
* Dùng khi bạn muốn viết script có ngữ cảnh Django đầy đủ (đã load settings, app, db...) - tiện cho việc gọi LLM,...
* <tên_lệnh> = tên file, Django sẽ tự dò các file trong your_app/management/commands/*.py. Bên trong đó có 1 lớp tên 
Command kế thừ BaseCommand, Django sẽ gọi lớp này khi chạy lệnh.
```python
class Command(BaseCommand):
    help = "Gọi LLM chào hỏi người dùng, cấu hình qua .env"

    def handle(self, *args, **kwargs):
        model = os.getenv("LLM_MODEL")
        llm = ChatOllama(model=model, temperature=0)

        prompt = "Bạn là trợ lý về phong thủy, hãy chào tôi bằng 1-2 câu tiếng Việt."

        t0 = time.time()
        res = llm.invoke(prompt)

        self.stdout.write(f"[{model}] {res.content} \nTook {time.time() - t0:.2f}s")
```
Chạy lệnh `python manage.py hello_llm`, ta sẽ thấy kết quả theo mẫu sau:
```
[llama3.1:8b] Chào bạn! Tôi rất vui được gặp và hỗ trợ bạn trong lĩnh vực phong thủy. Bạn cần tư vấn gì hôm nay? 
Took 19.87s
```

# Bài 2: LangChain căn bản (Prompt → Model → Output Parser)
## LangChain là gì?
* LangChain là bộ "lego" giúp bạn lắp ghép các bước làm việc với LLM: soạn prompt → gọi model → ép định dạng 
→ (tùy chọn) tìm tài liệu (RAG) → (tùy chọn) gọi tool → trả kết quả.
* Bạn lắp các bước trên bằng thứ gọi là "ống nối" LCEL (|). Ví dụ: `PromptTemplate | ChatModel | OutputParser`
* Một số mảnh "lego" phổ biến:
  * PromptTemplate: khuôn lời nhắc có biến ({question})
  * ChatModel: như chúng ta dùng là Ollama
  * OutputParser: ép định dạng trả về của model (JSON, text,...)
  * DocumentLoader + TextSplitter: đọc tài liệu và tách thành các đoạn nhỏ (cho RAG)
  * Embedding + VectorStore: chuyển đoạn văn thành vector rồi lưu để tìm theo ngữ nghĩa
  * Retriever: lấy top-k đoạn liên quan cho câu hỏi
  * Memory: lưu lịch sử hội thoại (tùy chọn)

  ...
* LangChain giúp chúng ta tập trung vào logic của AI mà không phải viết tay mọi kết nối (I/O, format,...) từ đầu.

**Mục tiêu bài 2**: tạo lệnh structured_qa - mẫu "Prompt → Model → Output Parser" (về sau có thể dùng để debug)
**Các bước thực hiện**
1. Tạo lệnh quản trị structured_qa trong copilot/management/commands/
2. Khai báo pydantic schema để ép định dạng trả về:
* Output Parser là lớp hậu xử lý trong LangChain. Dù đã nhắc LLM bằng format_instructions, parser vẫn cần để chuyển 
chuỗi đầu ra thành dữ liệu có cấu trúc (Pydantic model), kiểm tra & validate (kiểu, ràng buộc, thiếu trường), và ném 
lỗi sớm khi sai. Nhờ đó, pipeline ổn định và code phía sau dùng dữ liệu như object Python thay vì văn bản tự do.
```python
class FengshuiAnswer(BaseModel):
    # Dấu ... (gọi là ellipsis) ở đây là một giá trị đặc biệt từ module builtins của Python, đại diện cho việc
    # field này bắt buộc phải có giá trị (required) và không có giá trị mặc định (no default value).
    # Nếu muốn có default, bạn thay ... bằng giá trị cụ thể, ví dụ Field("Mặc định").
    answer: str = Field(..., description="Câu trả lời ngắn gọn bằng tiếng Viêt, ~80 từ")
    citations: List[str] = Field(
        default_factory=list,
        description="Các khái niệm/nguồn liên quan (ví dụ: ngũ hành, bát quái,...)"
    )
    confidence: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Độ tin cậy từ 0.0 đến 1.0"
    )
```
3. Tạo prompt template:
* Tạo format instruction dựa trên parser trước đó (chú ý đây chỉ là lời nhắc cho model thôi, về sau parser trong chain
vẫn phải kiểm tra và ép về chính xác định dạng chúng ta yêu cầu):
```python
class Command(BaseCommand):
    help = "Hỏi đáp phong thủy (LangChain + Ollama) có ép JSON"

    def add_arguments(self, parser):
        parser.add_argument("--q", dest="question", required=False,
                            default="Mệnh Kim hợp màu gì?", help="Câu hỏi tiếng Việt")

    def handle(self, *args, **opts):
        model = os.getenv("LLM_MODEL")
        llm = ChatOllama(model=model, temperature=0)

        # Tạo parser + format instructions
        parser = PydanticOutputParser(pydantic_object=FengshuiAnswer)
        format_instructions = parser.get_format_instructions()
```
* Tạo prompt template:
```python
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là chuyên gia phong thủy, hãy trả lời ngắn gọn, không bịa"
     " CHỈ TRẢ VỀ DỮ LIỆU THEO SCHEMA JSON được cung cấp"),
    ("human",
     "Câu hỏi: {question}\n\n"
     "Hãy tuân thủ nghiêm ngặt hướng dẫn định dạng sau:\n {format_instructions}")
])
```
4. Tạo chain và chạy:
```python
# Tạo chain:
chain = prompt | llm | parser

t0 = time.time()
try:
    # Đây chính là tác dụng của parser
    res: FengshuiAnswer = chain.invoke({
        "question": opts["question"],
        "format_instructions": format_instructions
    })
# Vì thừa, không đúng định dạng nên không thể parse được
except Exception as e:
    self.stderr.write(self.style.WARNING(f"Parse lỗi, thử fallback: {e}"))
    raw = (prompt | llm).invoke({
        "question": opts["question"],
        "format_instructions": format_instructions
    }).content

    match = re.search(r"\{.*\}", raw, re.DOTALL) # Tìm đoạn JSON dài nhất trong raw nhằm cắt ra khỏi văn bản thừa
    if not match:
        raise CommandError("Không tìm thấy JSON trong câu trả lời")
    res = parser.parse(match.group(0))

dt = time.time() - t0

self.stdout.write(json.dumps(res.dict(), ensure_ascii=False, indent=2))
self.stdout.write(self.style.SUCCESS(f"[{model}] took {dt:.2f}s"))
```
Test thử với lệnh:
```bash
python manage.py structured_qa --q "Nhà hướng Đông Nam hợp mệnh nào?"
````
Ta sẽ nhận được kết quả tương tự như sau:
```json
{
  "answer": "Nhà hướng Đông Nam hợp với người mệnh Mộc và Thủy",
  "citations": [
    "Ngũ hành",
    "Bát quái"
  ],
  "confidence": 0.8
}
[llama3.1:8b] took 95.44s
```

**Tổng kết bài 2:**
* Chúng ta đã tạo được lệnh hỏi đáp phong thủy có ép định dạng trả
* Chúng ta đã làm quen với các mảnh "lego" cơ bản của LangChain: PromptTemplate, ChatModel, OutputParser
* Chúng ta đã thấy được sức mạnh của OutputParser trong việc kiểm soát định dạng trả về, giúp pipeline ổn định hơn.
Về sau, chúng ta sẽ tiếp tục xây dựng dựa trên pipeline này để thêm RAG, tool, memory,...

# Bài 3: RAG v1 (Ingest → Split → Embed → Retrieve)
## RAG là gì?
* RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp LLM với hệ thống tìm kiếm tài liệu để cải thiện độ chính xác và
tính cập nhật của câu trả lời, giúp câu trả lời bám vào dữ kiện tài liệu của chúng ta, giảm bịa đặt.
* Ý tưởng chính: trước khi trả lời câu hỏi, LLM sẽ tìm kiếm các tài liệu liên quan trong kho dữ liệu (vector store) và sử dụng
chúng làm nguồn tham khảo để tạo câu trả lời.
* Quy trình RAG gồm các bước:
  1. Ingest: thu thập và nạp tài liệu vào hệ thống.
  2. Split: tách tài liệu thành các đoạn nhỏ để dễ quản lý và tìm kiếm.
  3. Embed: chuyển các đoạn văn bản thành vector (embedding) để lưu trữ và tìm kiếm theo ngữ nghĩa.
  4. Store: lưu trữ các vector trong cơ sở dữ liệu vector (vector store, ở đây chúng ta dùng Chroma).
  5. Retrieve: tìm kiếm và lấy các đoạn văn bản liên quan từ vector store dựa trên câu hỏi của người dùng.
  6. Generate: Kết hợp thông tin truy xuất được vào prompt gửi đến LLM (tăng cường). LLM tạo ra câu trả lời dựa trên dữ 
  liệu tăng cường, đảm bảo tính cập nhật và chính xác.

## Các bước thực hiện
1. Cài đặt các package cần thiết: langchain-chroma, chromadb, langchain-textsplitters

    Pull embedding model cho Ollama: `ollama pull nomic-embed-text`
2. Chuẩn bị tài liệu phong thủy:
* Tạo thư mục fengshui-copilot/data/corpus chưa các file tài liệu về phong thủy (md, txt, pdf,...)
* Tôi có cho sẵn một số tài liệu, bạn đọc có thể bổ sung thêm.

3. Tạo file cài đặt các tham số hệ thống:
* Tạo file copilot/rag/settings.py:
```python
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

CORPUS_DIR = Path(os.getenv("RAG_CORPUS_DIR", DATA_DIR / "corpus"))
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", DATA_DIR / "chroma"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "fengshui")

# Tham số split & retrieve
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 120))
TOP_K = int(os.getenv("RAG_TOP_K", 4))

# Model
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL")

# Đảm bảo thư mục tồn tại
def ensure_dirs():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
```
* CHUNK_SIZE: kích thước tối đa của mỗi chunk (đoạn văn bản nhỏ) sau khi split
* CHUNK_OVERLAP: độ chồng lắp giữa các chunk (giúp giữ ngữ cảnh) - tức là mỗi cặp chunk liền kề sẽ có một phần nội dung trùng nhau.
* TOP_K: số đoạn văn bản liên quan sẽ lấy ra để tăng cường cho LLM

4. Tạo file nạp (ingest) tài liệu:
* Tạo file copilot/rag/ingest.py
* Trước tiên, chúng ta viết 2 hàm load file:
```python
def _load_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _load_pdf_file(path:Path) -> str:
    text = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            text.append(page.extract_text() or "")
    except Exception:
        return ""

    return "\n".join(text)
```
* Viết hàm load_corpus để duyệt thư mục, đọc các file có đuôi phù hợp rồi trả về danh sách LangChain Document:
```python
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
                # thêm metadata để sau này hiện trích dẫn trong câu trả lời
            ))

    return docs
```
* Viết hàm chunk_documents để tách các Document lớn thành các đoạn nhỏ hơn có start_idx (vị trí bắt đầu trong văn bản gốc) để _make_id sau này tạo id ổn định:
```python
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    return splitter.split_documents(docs)

def _make_id(doc: Document) -> str:
    src = doc.metadata.get("source", "")
    start = doc.metadata.get("start_idx", None)
    return f"{src}::{start}"
```
* Viết hàm build_or_update_chroma để tạo hoặc cập nhật Chroma vector store:
```python
def build_or_update_chroma(chunks: List[Document], reset: bool = False) -> int:
    ensure_dirs()

    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Chroma vector store
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # Tạo id ổn định để tránh trùng lặp nếu ingest nhiều lần
    seen = set()
    docs_unique, ids_unique = [], []
    for doc in chunks:
        _id = _make_id(doc)
        if _id in seen:
            continue
        seen.add(_id)
        docs_unique.append(doc)
        ids_unique.append(_id)

    vs.add_documents(docs_unique, ids=ids_unique)

    return len(ids_unique)
```
* Viết hàm ingest_corpus tổng hợp các bước trên:
```python
def ingest_corpus(reset: bool = False) -> dict:
    ensure_dirs()
    docs = load_corpus(CORPUS_DIR)
    chunks = chunk_documents(docs) if docs else []
    n = build_or_update_chroma(chunks, reset=reset) if chunks else 0
    return {
        "files": len(docs),
        "chunks": len(chunks),
        "added": n,
        "corpus_dir": str(CORPUS_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "collection": COLLECTION_NAME,
    }
```
5. Tạo file truy xuất (retrieve) tài liệu:
* Tạo file copilot/rag/retrieve.py
```python
def get_retriever(top_k: int | None = None):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    return vs.as_retriever(search_kwargs={"k": top_k or TOP_K})
```
6. Tạo 2 lệnh quản trị để ingest và hỏi thử RAG:
* Tạo lệnh ingest_corpus trong copilot/management/commands/ingest_corpus.py
```python
def Command(BaseCommand):
    help = "Ingest tài liệu phong thuỷ vào Chroma (embed->split)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset", action="store_true",
            help="Xóa index cũ trước khi ingest"
        )

    def handle(self, *args, **opts):
        t0 = time.time()
        stats = ingest_corpus(reset=opts["reset"])
        dt = time.time() - t0

        self.stdout.write(self.style.SUCCESS(
            f"INGEST DONE in {dt:.2f}s | files={stats['files']} "
            f"chunks={stats['chunks']} added={stats['added']}\n"
            f"corpus={stats['corpus_dir']} | chroma={stats['chroma_dir']} "
            f"| collection={stats['collection']}"
        ))
```
* Tạo lệnh rag_ask trong copilot/management/commands/rag_ask.py
```python
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là trợ lý phong thuỷ. Trả lời ngắn gọn, dựa trên ngữ cảnh được cung cấp."
     " Nếu ngữ cảnh không đủ, hãy nói 'Tôi không chắc từ tài liệu hiện có.'"),
    ("human",
     "Câu hỏi: {question}\n\n"
     "Ngữ cảnh (có thể rút gọn):\n{context}\n\n"
     "Yêu cầu:\n- Trả lời 2–4 câu tiếng Việt, bám sát ngữ cảnh.\n"
     "- Liệt kê nguồn (tên file) đã dùng ở cuối câu trả lời.")
])

class Command(BaseCommand):
    help = "Hỏi–đáp với RAG (retriever + LLM), in câu trả lời kèm nguồn."

    def add_arguments(self, parser):
        parser.add_argument("--q", dest="question", required=True,
                            help="Câu hỏi phong thủy (tiếng Việt)")
        parser.add_argument("--k", dest="top_k", type=int, default=TOP_K,
                            help="Số đoạn trích dẫn lấy về (top k)")
        parser.add_argument("--model", default=LLM_MODEL,
                            help="Tên model Ollama")
        parser.add_argument("--temp", type=float, default=0.0)

    def handle(self, *args, **opts):
        q = opts["question"]
        k = opts["top_k"]
        model = opts["model"]
        temp = opts["temp"]

        # Lấy ngữ cảnh
        retriever = get_retriever(k)
        docs = retriever.get_relevant_documents(q)

        # Ghép đoạn trích tài liệu + nguồn để tạo ngữ cảnh
        ctx_lines = []
        used_files = set()

        for i, d in enumerate(docs):
            snippet = d.page_content.strip().replace("\n", " ")

            if len(snippet) > 500:
                snippet = snippet[:500] + "..."

            src = d.metadata.get("source", "")
            used_files.add(src)

            ctx_lines.append(f"[{i+1}] {snippet} (SOURCE: {src})")

        context = "\n\n".join(ctx_lines) if ctx_lines else "(Không có ngữ cảnh)"

        llm = ChatOllama(model=model, temperature=temp)
        chain = ANSWER_PROMPT | llm

        t0 = time.time()
        res = chain.invoke({
            "question": q,
            "context": context
        })
        dt = time.time() - t0

        answer = res.content.strip()
        sources = ", ".join(sorted(used_files)) if used_files else "Không có"
        self.stdout.write(f"[{model}] {answer}\n\n"
                          f"Sources: {sources}\n"
                          f"Took {dt:.2f}s")
```
* Test thử:
* Chạy lệnh ingest_corpus: `python manage.py ingest_corpus` hoặc `python manage.py ingest_corpus --reset` để xóa index cũ.
Output:
```

```
* Chạy lệnh rag_ask:
```bash
python manage.py rag_ask --q "Mệnh Kim hợp màu gì?"
python manage.py rag_ask --q "Nhà hướng Đông Nam hợp mệnh nào?" --k 6
```
Output:
```
[llama3.1:8b] Mệnh Kim hợp với màu sáng và nhẹ nhàng như màu trắng, vàng, ánh kim. Những màu này giúp cân bằng và hỗ trợ cho người mệnh Kim.

Nguồn:
1. phong_thuy_toan_tap.pdf
2. phong_thuy_thuc_hanh_trong_xay_dung_va_kien_truc_nha_o.pdf

Sources: phong_thuy_thuc_hanh_trong_xay_dung_va_kien_truc_nha_o.pdf, phong_thuy_toan_tap.pdf
Took 260.47s
```
```
[llama3.1:8b] Nhà hướng Đông Nam hợp với người mệnh Mộc và Thủy.

Nguồn: phong_thuy_toan_tap.pdf, phong_thuy_thuc_hanh_trong_xay_dung_va_kien_truc_nha_o.pdf

Sources: phong_thuy_thuc_hanh_trong_xay_dung_va_kien_truc_nha_o.pdf, phong_thuy_toan_tap.pdf
Took 299.03s
```
