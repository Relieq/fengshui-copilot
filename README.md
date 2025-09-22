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

    Pull embedding model cho Ollama: `ollama pull bge-m3` hoặc nomic-embed-text
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
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL") # Tôi dùng bge-m3
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
* Viết hàm chunk_documents để tách các Document lớn thành các đoạn nhỏ hơn có start_index (vị trí bắt đầu trong văn bản gốc) để _make_id sau này tạo id ổn định:
```python
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    return splitter.split_documents(docs)

def _make_id(doc: Document) -> str:
    src = doc.metadata.get("source", "")
    start = doc.metadata.get("start_index", None)
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
* Tạo file copilot/rag/retriever.py
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
        docs = retriever.invoke(q)

        # Ghép đoạn trích tài liệu + nguồn để tạo ngữ cảnh
        ctx_lines = []
        used_files = set()

        for i, d in enumerate(docs):
            snippet = d.page_content.strip().replace("\n", " ")

            # if len(snippet) > 500:
            #     snippet = snippet[:500] + "..."

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
* Chạy lệnh rag_ask:

Mọi người có thể xóa index trong \chroma xem trước kết quả như nào, sau đó hẵng ingest lại dữ liệu để thấy hiệu quả.
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

# Bài 4: Đánh giá RAG
* Mục tiêu: có các con số để chứng minh hệ RAG hoạt động.
* Evaluation set: 1 file jsonl, mỗi dòng gồm:
  * q: câu hỏi
  * sources: danh sách tên file trong corpus được coi là nguồn đúng (file-level)
  * ref: câu trả lời tham chiếu ngắn
* Trong bài này, chúng ta sẽ triển khai 3 cách đánh giá (2 đánh giá chất lượng truy vấn tài liệu, 1 đánh giá chất lượng 
câu trả lời):
  * Recall@k (file-level): % câu hỏi mà trong số tok-k chunk lấy về có ít nhất 1 chunk từ nguồn đúng.
  * MRR@k (Mean Reciprocal Rank (xếp hạng nghịch đảo trung bình)): trung bình của 1/rank với rank là vị trí chunk khớp 
  đúng đầu tiên với sources trong top-k (không khớp → 0).
  * Answer quality: chấm thô bằng "lexical" F1 (F1 score bản từ ngữ, so trùng từ giữa answer và ref). Có thể kèm model 
  cho điểm 0..1 dựa vào đúng/sai của nội dung.

_Lưu ý:_ Ở đây chúng ta chỉ mới triển khai ngang file-level, về sau có thể nâng cấp lên chunk-level (tính
trùng từ trong chunk lấy về với chunk đúng).

## Các bước thực hiện
### 1. Chuẩn bị dữ liệu đánh giá
* Lần này tôi có bổ sung thêm 4 file tài liệu nữa vào mục data (ban đầu 2). Ở bước này, chúng ta có thể nhờ các AI agent 
tạo giúp chúng ta file dữ liệu đánh giá qa.jsonl, mọi người có thể tham khảo prompt đơn giản sau:
```
Hiện tại tôi đang muốn tạo một tập dữ liệu đánh giá cho mô hình RAG trong project của tôi với tập eval set là 1 file .jsonl mỗi dòng gồm: 
- q: câu hỏi 
- sources: danh sách tên file trong corpus được coi là nguồn đúng (file-level) 
- ref (tùy chọn): câu trả lời tham chiếu ngắn 

Mẫu: 
{"q": "Mệnh Kim hợp màu gì?", "sources": ["ngu_hanh_co_ban.md"], "ref": "Mệnh Kim hợp trắng, xám, ánh kim; tương sinh Thổ như vàng, nâu."} 
{"q": "Nhà hướng Đông Nam hợp mệnh nào?", "sources": ["huong_nha_tom_tat.txt"], "ref": "Đông Nam thuộc Mộc, thường hợp mệnh Mộc và mệnh Hoả (tương sinh)."} 
{"q": "Ngũ hành gồm những yếu tố nào?", "sources": ["ngu_hanh_co_ban.md"], "ref": "Kim, Mộc, Thuỷ, Hoả, Thổ."} 

Về phần source, tôi có gửi cho bạn các nguồn như trên, bạn hãy load và đọc kĩ các file rồi tạo giúp tôi file jsonl phía trên. 
Tài liệu tìm được khá hạn chế, nếu có thể bạn hãy tự tìm kiếm, tải về, phân tích tài liệu rồi viết thêm vào file jsonl giúp tôi. 

Chú ý phải làm cho thật chính xác, phải có ít nhất 100 câu, để tôi có thể đánh giá kết quả mô hình của bản thân một cách có hiệu quả.
```
### 2. Tạo 2 lệnh đánh giá chất lượng retrieval tài liệu: Recall@k và MRR@k
* Tạo file copilot/rag/eval_retrieval.py
* Tạo hàm đọc file jsonl:
```python
def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
```
* Tạo lệnh:
```python
class Command(BaseCommand):
    help = "Đánh giá retrieval ở file-level với 2 phương pháp: Recall@k và MRR@k"

    def add_arguments(self, parser):
        parser.add_argument("--file", default=str(DATA_DIR / "eval" / "qa.jsonl"))
        parser.add_argument("--k", type=int, default=TOP_K)

    def handle(self, *args, **opts):
        eval_path = Path(opts["file"])
        k = opts["k"]

        data = list(read_jsonl(eval_path))
        if not data:
            raise CommandError(f"Eval set rỗng: {eval_path}")

        retriever = get_retriever(k)

        recalls, mrrs = [], []
        t0 = time.time()

        for i, item in enumerate(data):
            q = item["q"]
            gold = item.get("sources", "")
            docs = retriever.invoke(q)

            ranked_src = [d.metadata.get("source", "") for d in docs]
            hit = any(r in gold for r in ranked_src)
            recalls.append(hit)

            rr = 0.0
            for rank, src in enumerate(ranked_src):
                if src in gold:
                    rr = 1.0 / (rank + 1)
                    break
            mrrs.append(rr)

            self.stdout.write(
                f"[i] Q: {q}\n"
                f"gold: {gold}\n"
                f"got: {ranked_src}\n"
                f"Hit: {hit}, Reciprocal Rank (rr): {rr:.3f}\n"
            )

        dt = time.time() - t0
        self.stdout.write(self.style.SUCCESS(
              f"\nDone in {dt:.2f}s | k={k}\n"
              f"Recall@{k}: {mean(recalls):.3f} | MRR@{k}: {mean(mrrs):.3f} "
              f"{len(data)} câu."
        ))
```
* Test thử:
```bash
python manage.py eval_retrieval
```
Output:
```
...
[101] Q: Cửa phụ có cần lựa chọn theo tuổi gia chủ không?
gold: ['phong_thuy_thuc_hanh_trong_xay_dung_va_kien_truc_nha_o.pdf']
got: ['tu-vi-dau-so-toan-thu-tran-doan.pdf', 'TU_VI_THUC_HANH.pdf', 'phong_thuy_toan_tap.pdf', 'fengshui_phong_thuy_toan_tap.pdf']
Hit: False, Reciprocal Rank (rr): 0.000

Done in 4.93s | k=4
Recall@4: 0.337 | MRR@4: 0.097 101 câu.
```
### 3. Tạo lệnh đánh giá chất lượng câu trả lời dựa trên Lexical F1 (tùy chọn + LLM judge mini)
* Tạo file copilot/rag/eval_answer.py
* Viết hàm tokenize đơn giản chuyển câu thành tập các từ:
```python
def tokenize(s: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹ]+", (s or "").strip())
```
* Viết hàm tính F1 score:
```python
def f1_score(pred: str, ref: str) -> float:
    p = tokenize(pred)
    r = tokenize(ref)
    if not p or not r:
        return 0.0

    p_set = set(p)
    r_set = set(r)
    overlap = len(p_set & r_set)

    if overlap == 0:
        return 0.0

    precision = overlap / len(p_set)
    recall = overlap / len(r_set)
    return 2 * (precision * recall) / (precision + recall)
```
* Tạo prompt cho LLM đánh giá (chú ý phải "{{" chứ không phải "{")
```
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là chuyên gia, giám khảo về huyền học. Cho điểm 0..1 về ĐỘ CHÍNH XÁC so với câu tham chiếu. "
     "Chỉ chấm độ đúng (không chấm văn phong). Trả về đúng JSON: "
     '{{"score": <float>, "rationale": "<ngắn gọn>"}}'),
    ("human",
     "Câu hỏi: {question}\nTham chiếu: {ref}\nTrả lời: {pred}\n"
     "Chấm điểm và giải thích ngắn.")
])
```
* Tạo lệnh:
```python
class Command(BaseCommand):
    help = "Đánh giá chất lượng câu trả lời: F1 lexical + (tuỳ chọn) LLM judge mini."

    def add_arguments(self, parser):
        parser.add_argument("--file", default=str(DATA_DIR / "eval" / "qa.jsonl"))
        parser.add_argument("--k", type=int, default=TOP_K)
        parser.add_argument("--model", default=LLM_MODEL)
        parser.add_argument("--judge", action="store_true",
                            help="Bật chấm điểm bằng LLM")

    def handle(self, *args, **opts):
        eval_path = Path(opts["file"])
        k = opts["k"]
        model = opts["model"]
        use_judge = opts["judge"]

        data = list(read_jsonl(eval_path))
        if not data:
            raise CommandError(f"Eval set rỗng: {eval_path}")

        retriever = get_retriever(k)
        llm = ChatOllama(model=model)

        f1s, judge_scores = [], []
        t0 = time.time()

        for i, item in enumerate(data):
            q = item["q"]
            ref = item.get("ref", "").strip()

            docs = retriever.invoke(q)

            ctx_lines = []

            for j, d in enumerate(docs):
                snippet = d.page_content.strip().replace("\n", " ")

                # if len(snippet) > 500:
                #     snippet = snippet[:500] + "..."
                src = d.metadata.get("source", "")
                ctx_lines.append(f"[{j + 1}] {snippet} (SOURCE: {src})")

            context = "\n\n".join(ctx_lines) if ctx_lines else "(Không có ngữ cảnh)"

            pred = (ANSWER_PROMPT | llm).invoke({
                "question": q,
                "context": context
            }).content.strip()

            f1 = f1_score(pred, ref)
            f1s.append(f1)
            line = f"[{i+1}] Q: {q}\n REF: {ref}\n PRED: {pred}\n F1: {f1:.3f}"

            if use_judge:
                judge = (JUDGE_PROMPT | llm).invoke({
                    "question": q,
                    "ref": ref,
                    "pred": pred
                }).content.strip()

                m = re.search(r"\{.*\}", judge, re.DOTALL)
                if m:
                    try:
                        score = float(json.loads(m.group(0))["score"])
                    except Exception:
                        score = 0.0

                judge_scores.append(score)
                line += f" | Judge: {score:.3f} ({judge})"

            self.stdout.write(line + "\n")

        dt = time.time() - t0
        summary = f"\nDone in {dt:.2f}s | k={k}\n Avg F1: {mean(f1s):.3f}"
        if use_judge and judge_scores:
            summary += f" | Avg Judge: {mean(judge_scores):.3f}"
        self.stdout.write(self.style.SUCCESS(summary))
```
* Test thử:

Phần này nếu máy ai không đủ tài nguyên có thể xóa bớt nội dung trong file qa.jsonl để giảm số câu hỏi.
  * 
    ```bash
    python manage.py eval_answer
    ```
    Output:
    ```
    [1] Q: Quy luật Tương Sinh trong Ngũ Hành diễn ra theo thứ tự nào?
     REF: Thủy sinh Mộc; Mộc sinh Hỏa; Hỏa sinh Thổ; Thổ sinh Kim; Kim sinh Thủy.
     PRED: Quy luật Tương Sinh trong Ngũ Hành diễn ra theo thứ tự: Thủy sinh Mộc, Hỏa sinh Thổ, Thổ sinh Kim, Kim sinh Thủy.
    
    Nguồn:
    - phong_thuy_toan_tap.pdf
    - TU_VI_THUC_HANH.pdf
    - tu-vi-dau-so-toan-thu-tran-doan.pdf
     F1: 0.293
    [2] Q: Ngũ hành tương khắc theo thứ tự nào?
     REF: Thủy khắc Hỏa; Hỏa khắc Kim; Kim khắc Mộc; Mộc khắc Thổ; Thổ khắc Thủy.
     PRED: Ngũ hành tương khắc theo thứ tự: Kim khắc Hỏa, Thủy khắc Hỏa, Mộc khắc Thổ, Thổ khắc Thủy, Hỏa khắc Kim, Mộc khắc Kim.
    
    Nguồn:
    - TU_VI_THUC_HANH.pdf
    - tu-vi-dau-so-toan-thu-tran-doan.pdf
    - fengshui_phong_thuy_toan_tap.pdf
    - phong_thuy_thuc_hanh_trong_xay_dung_va_kien_truc_nha_o.pdf
     F1: 0.261
    ...
    ```
  *
    ```bash
    python manage.py eval_answer --judge
    ```
    Output:
    ```
    [1] Q: Quy luật Tương Sinh trong Ngũ Hành diễn ra theo thứ tự nào?
     REF: Thủy sinh Mộc; Mộc sinh Hỏa; Hỏa sinh Thổ; Thổ sinh Kim; Kim sinh Thủy.
     PRED: Quy luật Tương Sinh trong Ngũ Hành diễn ra theo thứ tự: Kim sinh Thủy, Thủy sinh Mộc, Mộc sinh Hỏa, Hỏa sinh Thổ, Thổ sinh Kim.
    
    Nguồn: phong_thuy_toan_tap.pdf, TU_VI_THUC_HANH.pdf, tu-vi-dau-so-toan-thu-tran-doan.pdf
     F1: 0.293 | Judge: 0.400 ({"score": 0.4, "rationale": "Danh sách thứ tự trong quy luật Tương Sinh được đưa ra là chính xác nhưng không theo trình tự vòng tròn Ngũ Hành (Kim, Thủy, Mộc, Hỏa, Thổ). Trình tự vòng tròn Ngũ Hành thường được sử dụng để miêu tả các mối quan hệ và quy luật của Ngũ Hành, vì vậy một danh sách thứ tự theo đúng vòng tròn có thể giúp người đọc dễ dàng nhận biết hơn về sự liên kết giữa các yếu tố trong Ngũ Hành."})
    
    Done in 77.75s | k=4
     Avg F1: 0.293 | Avg Judge: 0.400
    ```
* Chúng ta có thể dựa vào các con số này để đánh giá và cải thiện hệ thống RAG của mình, ví dụ:
  * Nếu Recall@k thấp, có thể do tài liệu không đủ hoặc quá trình embedding/retrieval chưa tốt.
  * Nếu F1 thấp, có thể do prompt chưa tốt hoặc LLM chưa hiểu đúng ngữ cảnh.
  * Dựa vào các câu hỏi cụ thể mà hệ thống trả lời sai để điều chỉnh prompt, thêm tài liệu, hoặc tinh chỉnh tham số:
    * Tăng k
    * Thay đổi chunk size/overlap
    * Thay embedding model

## LLM Provider linh hoạt
* Vấn đề: nhiều nơi gọi LLM (structured_qa, rag_ask, eval_answer, qa_graph). Nếu đổi provider (Ollama ↔ OpenRouter), sửa tay từng file sẽ dễ lỗi.
* Giải pháp: tạo một factory nhỏ get_chat(...) trả về model đã cấu hình sẵn dựa trên .env. Mọi lệnh chỉ from ... import get_chat và dùng.
* Lợi ích: DRY, đổi provider bằng sửa .env, không chạm code nghiệp vụ (RAG/graph giữ nguyên).
* Phạm vi: chỉ Chat model cho sinh câu trả lời/chấm điểm. Embeddings & Chroma vẫn dùng Ollama như Bài 3 (không đổi).
* Bạn có thể xem thống kê sử dụng trong trang activity của OpenRouter (nếu dùng OpenRouter làm provider).
![Xem thống kê sử dụng trong trang activity của OpenRouter](images/activity_dashboard_in_openrouter.jpeg)
Thực hành — Bật switch Ollama/OpenRouter
1) Cài gói (nếu chưa)
pip install -U langchain-openai openai langchain-ollama

2) Cập nhật .env
* Chú ý: sonoma-sky-alpha (OpenRouter) tôi dùng lúc này chỉ miễn phí trong thời gian nhất định, các bạn có thể tự tìm 
kiếm model khác phù hợp.
```
# Chọn 1:
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b

# Hoặc:
LLM_PROVIDER=openrouter
LLM_MODEL=openrouter/sonoma-sky-alpha
OPENROUTER_API_KEY=or-xxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=http://localhost:8000
OPENROUTER_APP_TITLE=fengshui-copilot-dev
```
3) Tạo factory: copilot/llm/provider.py
* Thêm vào llm/__init__.py
```python
def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v
```

Chỉ thêm 1 file nhỏ để tránh lặp code; không đụng gì tới RAG.
```python
# copilot/llm/provider.py
class ProviderError(RuntimeError): ...

def get_chat(temperature: float = 0.0):
    """
    Trả về Chat model đã cấu hình theo .env:
      - LLM_PROVIDER=ollama -> ChatOllama(model)
      - LLM_PROVIDER=openrouter -> ChatOpenAI(base_url=OpenRouter)
      - (tuỳ chọn) LLM_PROVIDER=openai -> ChatOpenAI (OpenAI gốc)
    """
    provider = (env("LLM_PROVIDER", "ollama") or "ollama")
    model = model or env("LLM_MODEL", "llama3.1:8b")

    if provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)

    if provider == "openrouter":
        api_key = env("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderError("Thiếu OPENROUTER_API_KEY trong .env")
        base_url = env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        headers = {
            "HTTP-Referer": env("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": env("OPENROUTER_APP_TITLE", "fengshui-copilot-dev"),
        }
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            default_headers=headers,
        )

    if provider == "openai":
        api_key = env("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("Thiếu OPENAI_API_KEY trong .env")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    raise ProviderError(f"LLM_PROVIDER không hỗ trợ: {provider}")
```
4) Sửa 4 lệnh để dùng factory (patch mẫu)
```
copilot/management/commands/structured_qa.py
- from langchain_ollama import ChatOllama
+ from ...llm.provider import get_chat
...
- llm = ChatOllama(model=model, temperature=temp)
+ llm = get_chat(temperature=temp)

copilot/management/commands/rag_ask.py
- from langchain_ollama import ChatOllama
+ from ...llm.provider import get_chat
...
- llm = ChatOllama(model=model, temperature=temp)
+ llm = get_chat(temperature=temp)

copilot/management/commands/eval_answer.py
- from langchain_ollama import ChatOllama
+ from ...llm.provider import get_chat
...
- llm = ChatOllama(model=model, temperature=0)
+ llm = get_chat(temperature=0)
...
- judge_raw = (JUDGE_PROMPT | llm).invoke(...)
+ judge_raw = (JUDGE_PROMPT | get_chat(temperature=0)).invoke(...)
```

Lưu ý: không đụng Bài 3 (ingest/retriever) — embeddings vẫn là OllamaEmbeddings(nomic-embed-text).

5) Kiểm thử nhanh (switch bằng .env, không sửa code)
```bash
python manage.py rag_ask --q "Nhà hướng Đông Nam hợp mệnh nào?"
```

Tôi có thử chạy lại lệnh `python manage.py eval_answer --judge` với OpenRouter (sonoma-sky-alpha), kết quả được được lưu 
vào file res_1.txt.

# Triển khai lại mục ingest:
Mục này được tạo ra vì lúc trước thật ra ở hàm _make_id() thuộc file ingest.py tôi có sơ suất ghi start_idx thay vì start_index, 
khiến tôi lầm tưởng rằng phần ingest không quá nặng (vì lấy start_idx - không có nên mặc định là None → trùng id → ít chunk). 

Lúc sau chỉnh sửa lại cho đúng thì thấy phần máy không chịu được nên từ mục này chúng ta sẽ triển khai lại như sau:
* Embedding: dùng Hugging Face (mặc định Endpoint API BAAI/bge-m3))
* Vector database: chuyển sang Supabase (Postgres + pgvector) với LangChain SupabaseVectorStore (vì máy tôi yếu + project 
chúng ta làm theo hướng production → Supabase được khuyến nghị là phù hợp hơn).
* Embedding model: BAAI/bge-m3, 1024 chiều (phù hợp tiếng Việt). Tham khảo thêm [tại đây](https://huggingface.co/BAAI/bge-m3).

## Bước 1: Tạo bảng và function trong Supabase
* Ở bước này bạn hãy tạo một project Supabase fengshui-copilot tại https://supabase.com/ (nếu chưa có).
* Sau đó trong Supabase dashboard → SQL Editor → Chọn Quickstarts "Langchain" ở mục Community, lúc đó một đoạn SQL query 
được tạo ra nhằm tạo bảng và function cần thiết cho LangChain SupabaseVectorStore.
![Quickstarts trong Supabase](images/quickstarts_supabase.jpeg)
* Chúng ta sẽ chỉnh sửa lại đoạn query này một chút:
```sql
-- Bật pgvector extension (nếu chưa)
create extension if not exists vector;

-- Tạo bảng documents
create table if not exists documents (
    id bigserial primary key, -- bigint, serial nghĩa là auto-increment
    uid text unique, -- id do ta tự tạo, tránh trùng lặp
    content text, -- tương ứng với Document.page_content
    metadata jsonb, -- json binary, hiệu quả hơn json thông thường, tương ứng với Document.metadata
    embedding vector(1024), -- bge-m3: 1024 dims
);

-- Tìm kiếm consine (trả similarity 0..1)
create or replace function match_documents(
    filter jsonb default '{}'::jsonb, -- bộ lọc metadata, ví dụ {"source": "file.pdf"} để chỉ tìm trong file.pdf
    match_count int default 4, -- top-k
    query_embedding vector(1024) default NULL -- embedding câu hỏi của người dùng
) returns table (
    id bigint,
    uid text,
    content text,
    metadata jsonb,
    embedding vector(1024),        -- <— QUAN TRỌNG: trả về vector dùng cho MMR
    similarity double precision
) language sql stable as $$ -- hàm viết bằng SQL thuần (không PL/pgSQL) 
    -- "stable" nghĩa là kết quả không thay đổi nếu input giống (tối ưu cache)
    select
        d.id,
        d.uid,
        d.content,
        d.metadata,
        d.embedding,
        1 - (d.embedding <=> query_embedding) as similarity -- cosine simlarity (1 - distance), 0..2
    from documents as d
    where d.metadata @> filter -- @>: contains
    order by d.embedding <=> query_embedding -- toán tử pgvector cho consine distance
    limit match_count;
$$;

-- Tạo index để tăng tốc tìm kiếm
create index if not exists documents_embedding_idx 
    -- Inverted file with flat: thuật toán approximate nearest neighbor (ANN) từ pgvector, nhanh cho vector search lớn.
    -- Thử tìm hiểu thì có vẻ giống k-means.
    on documents using ivfflat (embedding vector_cosine_ops)
    -- index trên cột embedding, dùng toán tử vector_cosine_ops để tối ưu cho consine distance.
    with (lists = 100); -- Số cụm trong ivfflat, càng lớn càng chính xác nhưng chậm hơn. Chúng ta sẽ xem xét lại sau.
    -- Quy tắc: lists ≈ sqrt(N) (N = số vectors), hoặc 1-4% của N. Ví dụ: N=10K → lists=100 (sqrt(10K)=100).
```

## Bước 2: Cấu hình môi trường
* Bổ sung các biến môi trường trong .env:
```
# Embeddings (Hugging Face)
EMBED_PROVIDER=hf_endpoint     # hoặc: hf_endpoint (nếu bạn có Endpoint/TEI riêng)
EMBEDDING_MODEL=BAAI/bge-m3
HUGGINGFACEHUB_API_TOKEN=hf_xxx...

# Supabase (server-side ONLY)
SUPABASE_URL=https://xxxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOi...
SUPABASE_TABLE=documents
SUPABASE_QUERY_NAME=match_documents
```

## Bước 3: Tạo factory embeddings (giống get_chat() ấy)
* HuggingFaceInferenceAPIEmbeddings / HuggingFaceEndpointEmbeddings được LangChain recommend cho Inference API/Endpoint. 
Supabase Python client khởi tạo bằng create_client(url, key).
* Tạo file copilot/llm/embeddings.py
```python
class EmbeddingProviderError(RuntimeError):...


def get_embeddings():
    provider = env("EMBED_PROVIDER", "hf_endpoint")
    model = env("EMBEDDING_MODEL", "BAAI/bge-m3")

    if provider == "ollama":
        return OllamaEmbeddings(model=model)

    if provider == "hf_inference":
        api_key = env("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise EmbeddingProviderError("Thiếu HUGGINGFACEHUB_API_TOKEN trong .env")
        return HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name=model)

    if provider == "hf_endpoint":
        api_key = env("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise EmbeddingProviderError("Thiếu HUGGINGFACEHUB_API_TOKEN trong .env")
        return HuggingFaceEndpointEmbeddings(
            model=model,
            task="feature-extraction",
            huggingfacehub_api_token=api_key
        )

    raise EmbeddingProviderError(f"EMBEDDING_PROVIDER không được hỗ trợ: {provider}")
```

## Bước 4: Chuyển Vector Store sang Supabase
* Chuyển rag thành module, tạo file supa.py:
```python
def get_supabase_client() -> client:
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Thiếu SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY trong .env")

    _SUPABASE_CLIENT = create_client(url, key)
    return _SUPABASE_CLIENT


def get_supabase_table_name() -> str:
    return os.getenv("SUPABASE_TABLE", "documents")


def get_supabase_query_name() -> str:
    # tên function tìm kiếm (match_documents) bạn đã tạo trong DB
    return os.getenv("SUPABASE_QUERY_NAME", "match_documents")
```
* Dùng SupabaseVectorStore + MMR cho retriever.py:
```python
def get_vectorstore():
    client = get_supabase_client()
    embeddings = get_embeddings()
    table = os.getenv("SUPABASE_TABLE", "documents")
    query = os.getenv("SUPABASE_QUERY_NAME", "match_documents")
    return SupabaseVectorStore(client=client, embedding=embeddings, table_name=table, query_name=query)

def get_retriever(top_k: int | None = None):
    vs = get_vectorstore()

    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(20, 5 * top_k)}
    )
```
* Tạo thêm hàm sanitize_text() để làm sạch text (bỏ ký tự không in được):
```python
# Vệ sinh, loại bỏ control char không mong muốn
def sanitize_text(s: str) -> str:
    if not s:
        return ""
    # chuẩn hoá xuống 1 khoảng trắng với control char; strip cho gọn
    s = _CONTROL_BAD.sub(" ", s)
    return s.strip()
```
* Chỉnh ingest thành theo batch:
```python
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
```
* Tạo RPC (Remote Procedure Call) trong Supabase để reset từ code:
```sql
create or replace function reset_documents()
returns void
language plpgsql
security definer -- đây là tùy chọn an ninh thiết lập rằng hàm sẽ chạy với quyền của người tạo hàm chứ không phải người gọi
as $$
begin
    truncate table documents restart identity;
end;
$$;
```
* Bạn có thể xem các hàm hiện có trong Database → Functions.
![function_supabase.jpeg](images/function_supabase.jpeg)
* Chỉnh tương ứng với file ingest_corpus.py:
```python
def handle(self, *args, **opts):
    if opts["reset"]:
        self.stdout.write("Xoá index cũ...")
        client = get_supabase_client()
        # table = get_supabase_table_name()
        # client.table(table).delete().neq("uid", None).execute()
        # self.stdout.write(self.style.WARNING(f"Đã reset bảng Supabase: {table}"))
        # gọi RPC thay vì delete()
        client.rpc("reset_documents").execute()
        self.stdout.write(self.style.SUCCESS("Đã TRUNCATE + RESTART IDENTITY cho bảng documents."))

    t0 = time.time()
    stats = ingest_corpus()
    dt = time.time() - t0

    self.stdout.write(self.style.SUCCESS(
        f"INGEST DONE in {dt:.2f}s | files={stats['files']} "
        f"chunks={stats['chunks']} added={stats['added']} deleted={stats['deleted']}\n"
        f"corpus={stats['corpus_dir']} "
        # f"| chroma={stats['chroma_dir']} | collection={stats['collection']}"
    ))
```
**Test thử:**
```bash
python manage.py ingest_corpus --reset
python manage.py rag_ask --q "Nhà hướng Đông Nam hợp mệnh nào?"
```
* Chạy lại lệnh eval_retrieval/eval_answer để xem kết quả thế nào.
```bash
python manage.py eval_retrieval --k 6
python manage.py eval_answer --judge --k 6
```
* Kết quả nhận được: Recall@6: 0.653 | MRR@6: 0.501 101 câu.
* Ở đây, chúng ta có thể thấy rằng, đúng là hiện tại việc ingest, retrieval đã nhẹ hơn rất nhiều, tuy nhiên vấn đề về chất 
lượng truy vấn vẫn KÉM.
* Tại sao? Các bạn có thể vào xem thử bảng documents trong Supabase để thấy rằng, nội dung mục content có nhiều chỗ bị 
lỗi font như vốn là "Ngũ hành" lại trở thành "Ngũ h{nh". Đây là lỗi phổ biến khi xử lý văn bản tiếng Việt, đặc biệt là từ các file PDF hoặc tài liệu được mã hóa không 
đúng (tương tự vấn đề control char đã xử lí lúc trước).
![vietnamese_text_problem.jpeg](images/vietnamese_text_problem.jpeg)
* 

# Bài 5: LangGraph – vòng lặp “trả lời → chấm điểm → (nếu kém) truy vấn lại”
## Khái niệm căn bản LangGraph
* LangGraph là thư viện để bạn "vẽ" đồ thị trạng thái cho quy trình nhiều bước với LLM:
  * State: 1 dict chứa các field chúng ta cần
  * Node: 1 hàm nhận state và trả về phần cập nhật state
  * Edge: đường đi giữa các node, có thể cố định (A → B) hoặc có điều kiện (A → B/C tùy dữ liệu trong state)
  * Loop: dùng cạnh có điều kiện để quay lại 1 node trước đó (ví dụ: chấm điểm thấp → quay lại truy vấn)
