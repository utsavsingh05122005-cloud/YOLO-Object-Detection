from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Enter the path of the video file
video_path = input("Enter the path of your video: ")

# Detect objects in the video
results = model.predict(
    source=video_path,
    show=True,
    save=True
)

print("Video object detection completed successfully!")

# python video_detection.py