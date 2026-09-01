from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


VECTORSTORE_PATH = "vectorstore"


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load FAISS database
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


# Ask question
question = input("Ask about an object: ")


# Search
results = vectorstore.similarity_search(
    question,
    k=1
)


print("\n--- JARVIS KNOWLEDGE ---\n")


for result in results:

    print(result.page_content)

    print("\n-----------------------\n")