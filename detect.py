from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("yolo26n.pt")

# Image path
image_path = "test.jpg"

# Open image
image = Image.open(image_path).convert("RGB")

# Detect objects
results = model(image)

# Process results
for result in results:

    print("\nDetected objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(f"Object: {class_name}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"Bounding Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        print("----------------------")

    # Create annotated image while preserving RGB
    annotated_image = result.plot(pil=True)
    

    # Show image
    annotated_image.show()