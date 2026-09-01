from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


TEXT_PATH = "documents/objects.txt"
VECTORSTORE_PATH = "vectorstore"


def create_vectorstore():

    # Read object information
    with open(TEXT_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    # Separate each OBJECT section
    sections = text.split("OBJECT:")

    documents = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        content = "OBJECT: " + section

        documents.append(
            Document(
                page_content=content
            )
        )

    print("Object information loaded successfully!")
    print("Number of objects:", len(documents))


    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating embeddings...")


    # Create FAISS database
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )


    # Save database
    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    print("Vector database created successfully!")


if __name__ == "__main__":
    create_vectorstore()