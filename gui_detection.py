import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from pathlib import Path
import threading
import cv2

# Load the YOLO model
model = YOLO("yolo26n.pt")
webcam_running = False

def detect_image():
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not image_path:
        return

    try:
        status_label.config(text="Status: Detecting objects in image...")
        root.update_idletasks()
        results = model.predict(source=image_path, save=True)
        result = results[0]

        output_path = Path(result.save_dir) / Path(image_path).name

        image = Image.open(output_path)
        image.thumbnail((550, 350))

        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo, text="")
        image_label.image = photo

        detected_objects = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = result.names[class_id]

            detected_objects.append(
                f"{object_name}: {confidence:.2f}"
            )

        results_label.config(
            text="Detected Objects:\n" + "\n".join(detected_objects)
            if detected_objects
            else "No objects detected."
        )

    except Exception as error:
        status_label.config(text="Status: Error occurred.")
        root.update_idletasks()
        messagebox.showerror("Error", str(error))


def detect_video():
    video_path = filedialog.askopenfilename(
        title="Select a Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if not video_path:
        return

    try:
        # Detect objects in the selected video
        model.predict(
            source=video_path,
            show=True,
            save=True
        )

        messagebox.showinfo(
            "Video Detection",
            "Video detection completed!\n"
            "The processed video is saved inside the runs/detect folder."
        )

    except Exception as error:
        messagebox.showerror("Error", str(error))


def detect_webcam():
    global webcam_running

    if webcam_running:
        return

    webcam_running = True
    status_label.config(text="Status: Webcam running")

    def webcam_loop():
        global webcam_running

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            webcam_running = False
            root.after(
                0,
                lambda: status_label.config(
                    text="Status: Could not open webcam"
                )
            )
            return

        try:
            while webcam_running:
                success, frame = cap.read()

                if not success:
                    break

                results = model.predict(frame, verbose=False)
                annotated_frame = results[0].plot()

                cv2.imshow("YOLO Webcam Detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            webcam_running = False

            root.after(
                0,
                lambda: status_label.config(
                    text="Status: Webcam stopped"
                )
            )

    threading.Thread(target=webcam_loop, daemon=True).start()


def stop_webcam():
    global webcam_running
    webcam_running = False
    status_label.config(text="Status: Stopping webcam...")

root = tk.Tk()

webcam_button = tk.Button(
    root,
    text="Start Webcam Detection",
    command=detect_webcam,
    font=("Arial", 11),
    padx=10,
    pady=8
)
webcam_button.pack(pady=8)
stop_button = tk.Button(
    root,
    text="Stop Webcam",
    command=stop_webcam,
    font=("Arial", 11),
    padx=10,
    pady=8
)
stop_button.pack(pady=8)

# Create the main window
root = tk.Tk()
root.title("YOLO Object Detection System")
root.geometry("700x650")

heading = tk.Label(
    root,
    text="YOLO Object Detection System",
    font=("Arial", 20, "bold")
)
heading.pack(pady=15)

image_button = tk.Button(
    root,
    text="Detect Objects in Image",
    command=detect_image,
    font=("Arial", 11),
    padx=10,
    pady=8
)
image_button.pack(pady=8)

video_button = tk.Button(
    root,
    text="Detect Objects in Video",
    command=detect_video,
    font=("Arial", 11),
    padx=10,
    pady=8
)
video_button.pack(pady=8)

image_label = tk.Label(
    root,
    text="Detected image will appear here."
)
image_label.pack(pady=10)

results_label = tk.Label(
    root,
    text="",
    font=("Arial", 11),
    justify="left"
)
results_label.pack(pady=10)

# Display application status
status_label = tk.Label(
    root,
    text="Status: Ready",
    font=("Arial", 10)
)
status_label.pack(pady=10)

root.mainloop()