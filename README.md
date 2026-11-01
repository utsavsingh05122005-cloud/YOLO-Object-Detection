# YOLO Object Detection System

A Python-based object detection project using **YOLO and Ultralytics** to identify objects in images, live webcam feeds, and recorded video files.

The project demonstrates how a pretrained deep learning model can detect objects and display bounding boxes with class labels and confidence scores.

## Features

- **Image Detection:** Detect objects in an image provided by the user.
- **Real-Time Webcam Detection:** Detect objects live using a webcam.
- **Video Detection:** Process recorded video files and save the output with detection annotations.
- **Confidence Scores:** Display the model's confidence for each detected object.
- **Detection Visualization:** Draw bounding boxes and object labels around recognized objects.

## Technologies Used

- Python
- YOLO
- Ultralytics
- OpenCV
- PyTorch

## Project Structure

```text
YOLO-Project/
│
├── yolo_detection.py       # Detection in custom images
├── webcam_detection.py    # Real-time webcam detection
├── video_detection.py     # Detection in video files
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Files excluded from Git
└── bus.jpg                # Sample image
```

The pretrained model file and generated detection results are kept locally and are not included in the repository.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/utsavsingh05122005-cloud/YOLO-Object-Detection.git
```

### 2. Open the project directory

```bash
cd YOLO-Object-Detection
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The pretrained model weights are downloaded automatically by Ultralytics when required and when an internet connection is available.

## How to Run

### 1. Image Detection

Run:

```bash
python yolo_detection.py
```

Enter the full path of the image when prompted.

Example:

```text
D:\YOLO-Project\bus.jpg
```

The program identifies objects and displays their class names and confidence scores. The annotated image is saved in the detection output directory.

### 2. Real-Time Webcam Detection

Run:

```bash
python webcam_detection.py
```

The program opens your default webcam and detects objects in real time.

**To stop:** Focus the detection window and press `Q`.

### 3. Video Detection

Run:

```bash
python video_detection.py
```

Enter the full path of a video file when prompted.

Example:

```text
D:\YOLO-Project\football.mp4
```

The program processes the video, displays the detections, and saves the annotated output.

**Note:** Replace the example path with the actual path of your video file.

## Sample Detection Results

The project has been tested with a sample bus image. The model detected:

- 4 persons
- 1 bus

Detection results may vary depending on the input image, video, lighting, and model confidence.

You can add screenshots of your own detection results to this section later.

## Learning Objectives

This project provides practical experience with:

- Using a pretrained object detection model
- Running inference with YOLO
- Processing images and video frames
- Working with webcam input
- Understanding object classes and confidence scores
- Organizing a Python project
- Using Git and GitHub for version control

## Future Improvements

- Build a graphical user interface.
- Add a feature to choose images through a file browser.
- Display detection statistics.
- Add support for selecting different YOLO models.
- Explore custom model training on a specific dataset.

## Author

**Utsav Singh**

Computer Science Engineering Student

GitHub: [utsavsingh05122005-cloud](https://github.com/utsavsingh05122005-cloud)

---

*This project was developed for learning and experimentation with computer vision and deep learning.*