from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from ultralytics import YOLO

from PIL import Image

import os
import uuid

from collections import Counter

from rag import (
    answer_question
)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# CONFIGURATION
# =========================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print()
print("=" * 60)
print("LOADING YOLO MODEL")
print("=" * 60)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo26n.pt"
)


model = YOLO(
    MODEL_PATH
)


print(
    "YOLO model loaded successfully!"
)


print()
print("YOLO classes:")
print(model.names)


# =========================================================
# IMAGE DETECTION
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    image_path = None

    detections = []

    grouped_detections = []

    total_objects = 0

    object_information = []


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


        detected_names = []


        # =================================================
        # PROCESS DETECTIONS
        # =================================================

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )


            confidence = float(
                box.conf[0]
            )


            class_name = model.names[
                class_id
            ]


            confidence_percent = round(
                confidence * 100,
                2
            )


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


            average_confidence = round(

                sum(confidences)
                /
                len(confidences),

                2

            ) if confidences else 0


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
        # GET INFORMATION FOR DETECTED OBJECTS
        # =================================================

        seen_objects = set()


        for object_name in detected_names:

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


            # -------------------------------------------------
            # Exact object lookup through local knowledge
            # -------------------------------------------------

            result_info = answer_question(
                object_name
            )


            information = result_info.get(
                "answer",
                "No information found."
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
        # SUMMARY
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


    # =====================================================
    # SUPPORT JSON REQUEST
    # =====================================================

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


    try:

        # =================================================
        # SEARCH LOCAL RAG
        # =================================================

        result = answer_question(
            question,
            k=3
        )


        answer = result.get(
            "answer",
            ""
        )


        sources = result.get(
            "sources",
            []
        )


        search_type = result.get(
            "type",
            "search"
        )


        # =================================================
        # RETURN RESPONSE
        # =================================================

        print()
        print(
            "SEARCH TYPE:",
            search_type
        )

        print(
            "SOURCES:",
            sources
        )

        print()
        print(
            "ANSWER:"
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


    except Exception as e:

        print(
            "RAG ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "answer":
                "An error occurred while "
                "searching the knowledge base.",

            "sources": [],

            "error":
                str(e)

        })


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "healthy",

        "application":
            "JARVIS",

        "yolo":
            "loaded",

        "rag":
            "local"

    })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(

        debug=False,

        host="0.0.0.0",

        port=5000

    )