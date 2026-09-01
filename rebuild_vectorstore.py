from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

OBJECTS_FILE = "objects_complete.txt"
VECTORSTORE_PATH = "vectorstore"

def load_object_documents():
    text = Path(OBJECTS_FILE).read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    documents = []
    for block in blocks:
        first_line = block.splitlines()[0].strip()
        if not first_line.startswith("OBJECT:"):
            continue

        object_name = first_line.split(":", 1)[1].strip().lower()
        documents.append(
            Document(
                page_content=block,
                metadata={"object": object_name, "source": OBJECTS_FILE}
            )
        )

    return documents

print("Loading object knowledge...")
documents = load_object_documents()
print(f"Loaded {len(documents)} objects.")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS vectorstore...")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(VECTORSTORE_PATH)

print("FAISS vectorstore rebuilt successfully.")
