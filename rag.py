import os
import pickle

from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# VECTORSTORE PATHS
# =========================================================

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
# LOAD VECTOR DATABASE
# =========================================================

def load_vectorstore():

    if not os.path.exists(DOCUMENTS_FILE):

        raise FileNotFoundError(
            "documents.pkl not found. "
            "Run rebuild_vectorstore.py first."
        )


    if not os.path.exists(VECTORIZER_FILE):

        raise FileNotFoundError(
            "vectorizer.pkl not found. "
            "Run rebuild_vectorstore.py first."
        )


    if not os.path.exists(MATRIX_FILE):

        raise FileNotFoundError(
            "matrix.pkl not found. "
            "Run rebuild_vectorstore.py first."
        )


    with open(
        DOCUMENTS_FILE,
        "rb"
    ) as file:

        documents = pickle.load(file)


    with open(
        VECTORIZER_FILE,
        "rb"
    ) as file:

        vectorizer = pickle.load(file)


    with open(
        MATRIX_FILE,
        "rb"
    ) as file:

        matrix = pickle.load(file)


    print(
        "Local vector database loaded successfully!"
    )

    print(
        "Number of documents:",
        len(documents)
    )


    return (
        documents,
        vectorizer,
        matrix
    )


# =========================================================
# LOAD DATABASE
# =========================================================

DOCUMENTS, VECTORIZER, MATRIX = load_vectorstore()


# =========================================================
# SEARCH KNOWLEDGE
# =========================================================

def search_knowledge(
    query,
    k=3
):

    query = (
        query
        .strip()
        .lower()
    )


    if not query:

        return []


    # -----------------------------------------------------
    # Convert question into local TF-IDF vector
    # -----------------------------------------------------

    query_vector = VECTORIZER.transform(
        [query]
    )


    # -----------------------------------------------------
    # Calculate similarity
    # -----------------------------------------------------

    similarities = cosine_similarity(
        query_vector,
        MATRIX
    )[0]


    # -----------------------------------------------------
    # Sort highest similarity first
    # -----------------------------------------------------

    ranked_indexes = similarities.argsort()[
        ::-1
    ]


    results = []


    for index in ranked_indexes[:k]:

        score = float(
            similarities[index]
        )


        # Ignore completely unrelated results
        if score <= 0:

            continue


        document = DOCUMENTS[index]


        results.append({

            "name":
                document["name"],

            "text":
                document["text"],

            "score":
                round(
                    score * 100,
                    2
                )

        })


    return results


# =========================================================
# EXACT OBJECT SEARCH
# =========================================================

def search_exact_object(
    object_name
):

    object_name = (
        object_name
        .strip()
        .lower()
    )


    for document in DOCUMENTS:

        if (
            document["name"].strip().lower()
            == object_name
        ):

            return document


    return None


# =========================================================
# SMART SEARCH
# =========================================================

def answer_question(
    question,
    k=3
):

    question = (
        question
        .strip()
        .lower()
    )


    if not question:

        return {
            "answer":
                "Please enter a question.",

            "sources": []
        }


    # =====================================================
    # FIRST: CHECK EXACT OBJECT NAME
    # =====================================================

    exact_match = search_exact_object(
        question
    )


    if exact_match:

        return {

            "answer":
                exact_match["text"],

            "sources":
                [exact_match["name"]],

            "type":
                "exact"

        }


    # =====================================================
    # SECOND: CHECK IF QUESTION CONTAINS OBJECT NAME
    # =====================================================

    for document in DOCUMENTS:

        object_name = (
            document["name"]
            .strip()
            .lower()
        )


        if (
            object_name in question
        ):

            return {

                "answer":
                    document["text"],

                "sources":
                    [object_name],

                "type":
                    "object"

            }


    # =====================================================
    # THIRD: SEMANTIC / TF-IDF SEARCH
    # =====================================================

    results = search_knowledge(
        question,
        k=k
    )


    if not results:

        return {

            "answer":
                "I could not find relevant "
                "information in my knowledge base.",

            "sources": [],

            "type":
                "none"

        }


    # -----------------------------------------------------
    # Build answer
    # -----------------------------------------------------

    answer_parts = []

    sources = []


    for result in results:

        answer_parts.append(
            result["text"]
        )


        sources.append(
            result["name"]
        )


    return {

        "answer":
            "\n\n".join(
                answer_parts
            ),

        "sources":
            sources,

        "type":
            "search"

    }