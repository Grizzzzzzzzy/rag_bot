from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import re
from collections import Counter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# CONFIGURATION
# =========================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

VECTORSTORE_PATH = os.path.join(
    BASE_DIR,
    "vectorstore"
)

OBJECTS_FILE = os.path.join(
    BASE_DIR,
    "documents",
    "objects_complete.txt"
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print()
print("=" * 60)
print("LOADING YOLO MODEL")
print("=" * 60)

model = YOLO(
    os.path.join(
        BASE_DIR,
        "yolo26n.pt"
    )
)

print("YOLO model loaded successfully!")

print()
print("YOLO classes:")
print(model.names)


# =========================================================
# LOAD RAG VECTOR DATABASE
# =========================================================

print()
print("=" * 60)
print("LOADING RAG VECTOR DATABASE")
print("=" * 60)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


print("RAG vector database loaded successfully!")


# =========================================================
# LOAD OBJECT KNOWLEDGE
# =========================================================

def load_object_knowledge():

    knowledge = {}

    print()
    print("=" * 60)
    print("LOADING OBJECT KNOWLEDGE")
    print("=" * 60)

    print(
        "Looking for:",
        OBJECTS_FILE
    )

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if not os.path.exists(OBJECTS_FILE):

        print(
            "ERROR: Object knowledge file not found!"
        )

        return knowledge


    print(
        "Object knowledge file found!"
    )


    # -----------------------------------------------------
    # READ FILE
    # -----------------------------------------------------

    try:

        with open(
            OBJECTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read()

    except Exception as e:

        print(
            "ERROR READING OBJECT FILE:",
            e
        )

        return knowledge


    # -----------------------------------------------------
    # SPLIT OBJECTS
    # -----------------------------------------------------

    blocks = [
        block.strip()
        for block in re.split(
            r"\n\s*\n",
            content
        )
        if block.strip()
    ]


    print(
        "Number of blocks found:",
        len(blocks)
    )


    # -----------------------------------------------------
    # PROCESS OBJECT BLOCKS
    # -----------------------------------------------------

    for block in blocks:

        lines = block.splitlines()

        object_name = None


        for line in lines:

            line = line.strip()

            if line.upper().startswith("OBJECT:"):

                object_name = (
                    line
                    .split(
                        ":",
                        1
                    )[1]
                    .strip()
                    .lower()
                )

                break


        # -------------------------------------------------
        # SAVE OBJECT
        # -------------------------------------------------

        if object_name:

            knowledge[object_name] = block


    # -----------------------------------------------------
    # PRINT LOADED OBJECTS
    # -----------------------------------------------------

    print(
        "Loaded exact knowledge for",
        len(knowledge),
        "objects."
    )


    print()
    print("Available object knowledge:")

    for object_name in knowledge:

        print(
            " -",
            object_name
        )


    print(
        "=" * 60
    )

    return knowledge


# =========================================================
# LOAD KNOWLEDGE ONCE
# =========================================================

OBJECT_KNOWLEDGE = load_object_knowledge()


# =========================================================
# FIND OBJECT IN QUESTION
# =========================================================

def find_objects_in_question(question):

    """
    Finds exact object names from the question.

    Example:

    "what is bus?"
        -> ["bus"]

    "tell me about the truck"
        -> ["truck"]

    "what is a traffic light?"
        -> ["traffic light"]

    The longest object names are checked first so that
    multi-word objects work correctly.
    """

    question = (
        question
        .strip()
        .lower()
    )


    # -----------------------------------------------------
    # Normalize punctuation
    # -----------------------------------------------------

    normalized_question = re.sub(
        r"[^\w\s]",
        " ",
        question
    )

    normalized_question = re.sub(
        r"\s+",
        " ",
        normalized_question
    ).strip()


    found_objects = []


    # -----------------------------------------------------
    # Sort longest names first
    # -----------------------------------------------------

    object_names = sorted(
        OBJECT_KNOWLEDGE.keys(),
        key=len,
        reverse=True
    )


    # -----------------------------------------------------
    # Search exact object names
    # -----------------------------------------------------

    for object_name in object_names:

        pattern = (
            r"\b"
            + re.escape(object_name)
            + r"\b"
        )


        if re.search(
            pattern,
            normalized_question
        ):

            found_objects.append(
                object_name
            )


    # -----------------------------------------------------
    # Remove duplicates
    # -----------------------------------------------------

    found_objects = list(
        dict.fromkeys(
            found_objects
        )
    )


    return found_objects


# =========================================================
# GET EXACT OBJECT INFORMATION
# =========================================================

def get_object_information(object_name):

    object_name = (
        object_name
        .strip()
        .lower()
    )


    # -----------------------------------------------------
    # EXACT MATCH
    # -----------------------------------------------------

    if object_name in OBJECT_KNOWLEDGE:

        print(
            "Exact knowledge found:",
            object_name
        )

        return OBJECT_KNOWLEDGE[
            object_name
        ]


    # -----------------------------------------------------
    # NORMALIZED MATCH
    # -----------------------------------------------------

    normalized_name = (
        object_name
        .replace(
            "_",
            " "
        )
        .strip()
    )


    if normalized_name in OBJECT_KNOWLEDGE:

        print(
            "Normalized knowledge found:",
            normalized_name
        )

        return OBJECT_KNOWLEDGE[
            normalized_name
        ]


    # -----------------------------------------------------
    # NO INFORMATION
    # -----------------------------------------------------

    print(
        "No knowledge found:",
        object_name
    )

    return (
        "No specific information found "
        "for this object."
    )


# =========================================================
# RAG SEARCH
# =========================================================

def search_knowledge(
    query,
    k=3
):

    try:

        results = vectorstore.similarity_search(
            query,
            k=k
        )

        return results

    except Exception as e:

        print(
            "RAG SEARCH ERROR:",
            e
        )

        return []


# =========================================================
# HOME / IMAGE DETECTION
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    image_path = None

    detections = []

    grouped_detections = []

    object_information = []

    total_objects = 0


    # =====================================================
    # GET REQUEST
    # =====================================================

    if request.method == "GET":

        return render_template(
            "index.html",
            image_path=None,
            detections=[],
            grouped_detections=[],
            object_information=[],
            total_objects=0
        )


    # =====================================================
    # CHECK IMAGE
    # =====================================================

    if "image" not in request.files:

        return render_template(
            "index.html",
            image_path=None,
            detections=[],
            grouped_detections=[],
            object_information=[],
            total_objects=0,
            error="No image was uploaded."
        )


    file = request.files["image"]


    # =====================================================
    # CHECK FILENAME
    # =====================================================

    if file.filename == "":

        return render_template(
            "index.html",
            image_path=None,
            detections=[],
            grouped_detections=[],
            object_information=[],
            total_objects=0,
            error="Please select an image."
        )


    try:

        # =================================================
        # SAVE IMAGE
        # =================================================

        extension = os.path.splitext(
            file.filename
        )[1].lower()


        if not extension:

            extension = ".jpg"


        unique_name = (
            str(uuid.uuid4())
            + extension
        )


        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_name
        )


        file.save(
            file_path
        )


        # =================================================
        # START ANALYSIS
        # =================================================

        print()
        print("=" * 60)
        print("JARVIS IMAGE ANALYSIS")
        print("=" * 60)


        # =================================================
        # OPEN IMAGE
        # =================================================

        image = Image.open(
            file_path
        ).convert("RGB")


        # =================================================
        # YOLO DETECTION
        # =================================================

        results = model(
            image
        )


        result = results[0]


        # =================================================
        # GET DETECTIONS
        # =================================================

        detected_names = []


        for box in result.boxes:

            # ---------------------------------------------
            # CLASS ID
            # ---------------------------------------------

            class_id = int(
                box.cls[0]
            )


            # ---------------------------------------------
            # CONFIDENCE
            # ---------------------------------------------

            confidence = float(
                box.conf[0]
            )


            # ---------------------------------------------
            # OBJECT NAME
            # ---------------------------------------------

            class_name = model.names[
                class_id
            ]


            confidence_percent = round(
                confidence * 100,
                2
            )


            # ---------------------------------------------
            # SAVE DETECTION
            # ---------------------------------------------

            detections.append({

                "name":
                    class_name,

                "confidence":
                    confidence_percent

            })


            detected_names.append(
                class_name.lower()
            )


            print(
                f"Detected: {class_name} "
                f"({confidence_percent}%)"
            )


        # =================================================
        # TOTAL OBJECTS
        # =================================================

        total_objects = len(
            detections
        )


        # =================================================
        # GROUP OBJECTS
        # =================================================

        object_counts = Counter(
            detected_names
        )


        for object_name, count in (
            object_counts.items()
        ):

            confidences = [

                item["confidence"]

                for item in detections

                if item["name"].lower()
                == object_name

            ]


            if confidences:

                average_confidence = round(

                    sum(confidences)
                    /
                    len(confidences),

                    2

                )

            else:

                average_confidence = 0


            grouped_detections.append({

                "name":
                    object_name,

                "count":
                    count,

                "confidence":
                    average_confidence

            })


        # =================================================
        # SORT OBJECTS
        # =================================================

        grouped_detections.sort(

            key=lambda x: x["count"],

            reverse=True

        )


        # =================================================
        # EXACT KNOWLEDGE FOR DETECTED OBJECTS
        # =================================================

        seen_objects = set()


        for object_name in detected_names:

            # Don't repeat same object
            if object_name in seen_objects:

                continue


            seen_objects.add(
                object_name
            )


            print()
            print(
                "Getting information for:",
                object_name
            )


            information = (
                get_object_information(
                    object_name
                )
            )


            object_information.append({

                "name":
                    object_name,

                "information":
                    information

            })


        # =================================================
        # CREATE ANNOTATED IMAGE
        # =================================================

        annotated_image = result.plot()


        output_name = (
            "result_"
            + unique_name
        )


        output_path = os.path.join(

            app.config["UPLOAD_FOLDER"],

            output_name

        )


        Image.fromarray(

            annotated_image[..., ::-1]

        ).save(
            output_path
        )


        # =================================================
        # IMAGE URL
        # =================================================

        image_path = (
            "/static/uploads/"
            + output_name
        )


        # =================================================
        # PRINT SUMMARY
        # =================================================

        print()
        print("=" * 60)

        print(
            "TOTAL OBJECTS:",
            total_objects
        )

        print(
            "UNIQUE OBJECTS:",
            len(object_counts)
        )

        print(
            "=" * 60
        )


    except Exception as e:

        print()
        print(
            "IMAGE PROCESSING ERROR:",
            e
        )


        return render_template(

            "index.html",

            image_path=None,

            detections=[],

            grouped_detections=[],

            object_information=[],

            total_objects=0,

            error=str(e)

        )


    # =====================================================
    # SEND DATA TO HTML
    # =====================================================

    return render_template(

        "index.html",

        image_path=image_path,

        detections=detections,

        grouped_detections=grouped_detections,

        object_information=object_information,

        total_objects=total_objects

    )


# =========================================================
# RAG QUESTION ROUTE
# =========================================================

@app.route(
    "/rag",
    methods=["POST"]
)
def rag():

    print()
    print("=" * 60)
    print("JARVIS RAG SYSTEM")
    print("=" * 60)


    # =====================================================
    # GET QUESTION
    # =====================================================

    question = request.form.get(
        "question",
        ""
    ).strip()


    # -----------------------------------------------------
    # SUPPORT JSON
    # -----------------------------------------------------

    if (
        not question
        and request.is_json
    ):

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )


        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()


    print(
        "QUESTION:",
        question
    )


    # =====================================================
    # EMPTY QUESTION
    # =====================================================

    if not question:

        return jsonify({

            "success": False,

            "answer":
                "Please enter a question.",

            "sources": []

        })


    # =====================================================
    # EXACT OBJECT DETECTION
    # =====================================================

    matched_objects = (
        find_objects_in_question(
            question
        )
    )


    print(
        "OBJECTS FOUND IN QUESTION:",
        matched_objects
    )


    # =====================================================
    # EXACT OBJECT SEARCH
    # =====================================================

    if matched_objects:

        print()
        print(
            "EXACT OBJECT SEARCH MODE"
        )


        answer_parts = []

        sources = []


        for object_name in matched_objects:

            information = (
                get_object_information(
                    object_name
                )
            )


            # ---------------------------------------------
            # Add information
            # ---------------------------------------------

            if information:

                answer_parts.append(
                    information
                )


            # ---------------------------------------------
            # Add source
            # ---------------------------------------------

            sources.append(
                object_name
            )


        # -------------------------------------------------
        # FINAL EXACT ANSWER
        # -------------------------------------------------

        answer = "\n\n".join(
            answer_parts
        )


        print()
        print(
            "EXACT RAG ANSWER:"
        )

        print(
            answer
        )


        print(
            "=" * 60
        )


        return jsonify({

            "success": True,

            "answer":
                answer,

            "sources":
                sources

        })


    # =====================================================
    # NORMAL RAG SEARCH
    # =====================================================

    print()
    print(
        "NO EXACT OBJECT FOUND."
    )

    print(
        "Using FAISS similarity search..."
    )


    documents = search_knowledge(
        question,
        k=3
    )


    # =====================================================
    # NO RESULTS
    # =====================================================

    if not documents:

        print(
            "No relevant documents found."
        )


        return jsonify({

            "success": False,

            "answer":
                "I could not find relevant "
                "information in my knowledge base.",

            "sources": []

        })


    # =====================================================
    # BUILD FAISS ANSWER
    # =====================================================

    answer_parts = []

    sources = []


    for document in documents:

        # ---------------------------------------------
        # DOCUMENT TEXT
        # ---------------------------------------------

        text = (
            document.page_content
            .strip()
        )


        if (
            text
            and
            text not in answer_parts
        ):

            answer_parts.append(
                text
            )


        # ---------------------------------------------
        # SOURCE
        # ---------------------------------------------

        metadata = (
            document.metadata
            or {}
        )


        source = (
            metadata.get("source")
            or
            metadata.get("file")
            or
            metadata.get("filename")
        )


        if (
            source
            and
            source not in sources
        ):

            sources.append(
                source
            )


    # =====================================================
    # FINAL FAISS ANSWER
    # =====================================================

    if answer_parts:

        answer = "\n\n".join(
            answer_parts
        )

    else:

        answer = (
            "Relevant information was found, "
            "but no readable text was available."
        )


    # =====================================================
    # PRINT RESULT
    # =====================================================

    print()
    print(
        "RAG ANSWER:"
    )

    print(
        answer
    )


    print()
    print(
        "SOURCES:",
        sources
    )


    print(
        "=" * 60
    )


    # =====================================================
    # RETURN JSON
    # =====================================================

    return jsonify({

        "success": True,

        "answer":
            answer,

        "sources":
            sources

    })


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )