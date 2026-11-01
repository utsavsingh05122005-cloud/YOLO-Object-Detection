from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Detect objects in the sample image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True
)

# Display detected objects
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        object_name = result.names[class_id]

        print(f"Object: {object_name}, Confidence: {confidence:.2f}")

print("Object detection completed successfully!")