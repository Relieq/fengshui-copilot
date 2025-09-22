from email.quoprimime import decode

from langchain_community.document_loaders import PyMuPDFLoader
from pypdf import PdfReader
import re, unicodedata


# Tạo hàm giải mã pdf mã hóa:
# '{', '}', '~', '|' lần lượt tương ứng là 'à', 'â', 'ã', 'á'
def decode_pdf_text(s: str) -> str:
    if not s:
        return ""
    s = (s.replace("{", "à").replace("}", "â")
         .replace("~", "ã").replace("|", "á"))
    return s

print("Loading PDF with PyMuPDFLoader...") # Cái này có vẻ tốt hơn
docs = PyMuPDFLoader("data/corpus/fengshui_phong_thuy_toan_tap.pdf.pdf").load()
# print("\n".join(d.page_content for d in docs))
content = decode_pdf_text("\n".join(d.page_content for d in docs))
print(content)

# print("\nLoading PDF with pypdf...")
# reader = PdfReader("data/corpus/fengshui_phong_thuy_toan_tap.pdf")
# texts = []
# for page in reader.pages:
#     texts.append(page.extract_text())
# print("\n".join(texts))

# print("\nLoading PDF with PDFMinerLoader...")
# loader = PDFMinerLoader("data/corpus/fengshui_phong_thuy_toan_tap.pdf")
# docs = loader.load()
# print("\n".join(d.page_content for d in docs))
