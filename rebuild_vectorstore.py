import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# PATHS
# =========================================================

OBJECTS_FILE = os.path.join(
    BASE_DIR,
    "documents",
    "objects_complete.txt"
)

VECTORSTORE_PATH = os.path.join(
    BASE_DIR,
    "vectorstore"
)

DOCUMENTS_FILE = os.path.join(
    VECTORSTORE_PATH,
    "documents.pkl"
)

VECTORIZER_FILE = os.path.join(
    VECTORSTORE_PATH,
    "vectorizer.pkl"
)

MATRIX_FILE = os.path.join(
    VECTORSTORE_PATH,
    "matrix.pkl"
)


# =========================================================
# CREATE VECTORSTORE DIRECTORY
# =========================================================

os.makedirs(
    VECTORSTORE_PATH,
    exist_ok=True
)


# =========================================================
# LOAD OBJECT DOCUMENTS
# =========================================================

def load_documents():

    print()
    print("=" * 60)
    print("LOADING OBJECT KNOWLEDGE")
    print("=" * 60)

    print(
        "File:",
        OBJECTS_FILE
    )

    if not os.path.exists(OBJECTS_FILE):

        raise FileNotFoundError(
            f"Object file not found: {OBJECTS_FILE}"
        )


    with open(
        OBJECTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()


    # -----------------------------------------------------
    # Split knowledge into individual object blocks
    # -----------------------------------------------------

    blocks = [
        block.strip()
        for block in content.split("\n\n")
        if block.strip()
    ]


    documents = []


    for block in blocks:

        lines = block.splitlines()

        object_name = None


        for line in lines:

            line = line.strip()

            if line.upper().startswith("OBJECT:"):

                object_name = (
                    line
                    .split(":", 1)[1]
                    .strip()
                    .lower()
                )

                break


        if object_name:

            documents.append({
                "name": object_name,
                "text": block
            })


    print(
        "Objects loaded:",
        len(documents)
    )


    for document in documents:

        print(
            " -",
            document["name"]
        )


    print("=" * 60)

    return documents


# =========================================================
# BUILD TF-IDF VECTOR DATABASE
# =========================================================

def build_vectorstore():

    documents = load_documents()


    if not documents:

        raise ValueError(
            "No object documents found."
        )


    # -----------------------------------------------------
    # Extract text
    # -----------------------------------------------------

    texts = [
        document["text"]
        for document in documents
    ]


    # -----------------------------------------------------
    # Create local TF-IDF embeddings
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("CREATING LOCAL TF-IDF EMBEDDINGS")
    print("=" * 60)


    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )


    matrix = vectorizer.fit_transform(
        texts
    )


    print(
        "Embedding matrix shape:",
        matrix.shape
    )


    # -----------------------------------------------------
    # Save documents
    # -----------------------------------------------------

    with open(
        DOCUMENTS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            documents,
            file
        )


    # -----------------------------------------------------
    # Save vectorizer
    # -----------------------------------------------------

    with open(
        VECTORIZER_FILE,
        "wb"
    ) as file:

        pickle.dump(
            vectorizer,
            file
        )


    # -----------------------------------------------------
    # Save matrix
    # -----------------------------------------------------

    with open(
        MATRIX_FILE,
        "wb"
    ) as file:

        pickle.dump(
            matrix,
            file
        )


    print()
    print("=" * 60)
    print("VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(
        "Documents:",
        DOCUMENTS_FILE
    )

    print(
        "Vectorizer:",
        VECTORIZER_FILE
    )

    print(
        "Matrix:",
        MATRIX_FILE
    )

    print("=" * 60)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    build_vectorstore()