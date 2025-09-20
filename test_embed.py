from dotenv import load_dotenv

from copilot.llm.embeddings import get_embeddings


# load .env
load_dotenv(".env")

emb = get_embeddings()
print(emb.embed_documents(["xin chào", "mệnh kim hợp màu gì"]))
# → nhận list 2 vector (mỗi vector 1024 float)

