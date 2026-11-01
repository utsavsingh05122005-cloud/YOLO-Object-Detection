from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Start object detection using the webcam
results = model.predict(
    source=0,
    show=True,
    stream=True
)

# Process webcam frames
for result in results:
    pass