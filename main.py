import cv2 as cv
import numpy as np
from ultralytics import YOLO

# Load YOLO model
np_model = YOLO('yolov8n.pt')

# Path to the video file
video_path = 'assets/sample.mp4'

# Open video file
video = cv.VideoCapture(video_path)

# Check if the video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

# Define codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('./results.mp4', fourcc, 20.0, size)

# Read and process frames
while True:
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Detect & track objects
    results = np_model.track(frame, persist=True)

    # Plot results
    composed = results[0].plot()
    
    # Save video
    out.write(composed)

# Release resources
out.release()
video.release()
cv.destroyAllWindows()
