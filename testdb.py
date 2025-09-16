import chromadb

# Kết nối client (thay path cho đúng)
client = chromadb.PersistentClient(path="data/chroma")

# Lấy collection
collection = client.get_collection("fengshui")

# Lấy tất cả (hoặc limit nếu nhiều)
results = collection.get(
    include=["documents", "metadatas"],
    limit=50  # Giới hạn để tránh overload nếu collection lớn
)

# In ra theo từng chunk để dễ phân biệt
for i in range(len(results['ids'])):
    doc_id = results['ids'][i]
    doc_text = results['documents'][i][:500] + "..."  # Truncate để ngắn gọn
    metadata = results['metadatas'][i] if results['metadatas'][i] else {}  # Dict metadata

    print(f"\n--- Chunk ID: {doc_id} ---")
    print(f"Document: {doc_text}")
    print(f"Source file: {metadata.get('source', 'N/A')}")  # Giả sử key là 'source'
    print(f"Page/Section: {metadata.get('page', metadata.get('section', 'N/A'))}")
    print("Full metadata:", metadata)  # Để xem hết keys có sẵn