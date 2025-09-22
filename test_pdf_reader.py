from langchain_community.document_loaders import PyMuPDFLoader
from pypdf import PdfReader
import re, unicodedata

print("Loading PDF with PyMuPDFLoader...") # Cái này có vẻ tốt hơn
docs = PyMuPDFLoader("data/corpus/fengshui_phong_thuy_toan_tap.pdf").load()
print("\n".join(d.page_content for d in docs))

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
