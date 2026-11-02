from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Ask the user for an image path
image_path = input("Enter the path of your image: ")

# Detect objects in the image
results = model.predict(source=image_path, save=True)

# Display detected objects
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        object_name = result.names[class_id]

        print(f"Object: {object_name}, Confidence: {confidence:.2f}")
        
print("Object detection completed successfully!")

# python yolo_detection.py