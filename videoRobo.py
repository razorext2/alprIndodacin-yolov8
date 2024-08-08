import cv2 as cv
import os
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="MMkdGQyPw2nS3ttNfaoZ")
project = rf.workspace().project("platedetector-zkpmk")
model = project.version("11").model

# Path to the input video
input_video_path = '2.mp4'

# Open a connection to the input video file
video = cv.VideoCapture(input_video_path)

# Check if the video file opened successfully
if not video.isOpened():
    print(f"Error: Could not open video file {input_video_path}.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
output_path = './outputs/output.mp4'
if not os.path.exists('./outputs'):
    os.makedirs('./outputs/img')
out = cv.VideoWriter(output_path, fourcc, 10.0, (int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT))))

# Set to keep track of saved detections
saved_detections = set()

# Function to save detected objects with bounding boxes
def save_detected_objects(frame, predictions, frame_count):
    global saved_detections
    for prediction in predictions['predictions']:
        x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        confidence = prediction['confidence']
        class_name = prediction['class']
        track_id = prediction.get('track_id')  # Get track ID if available

        # Only draw boxes with confidence above 0.50
        if confidence >= 0.50:
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Create a unique identifier for the detection using track_id
            detection_id = track_id if track_id is not None else (x1, y1, x2, y2, class_name)

            # Save the cropped detected object if not already saved
            if detection_id not in saved_detections:
                detected_obj = frame[y1:y2, x1:x2]
                obj_path = f'./outputs/img/frame_{frame_count}_{class_name}_{confidence:.2f}.jpg'
                cv.imwrite(obj_path, detected_obj)
                saved_detections.add(detection_id)
                print(f"Saved detection: {detection_id}")

# Read frames and perform inference
frame_count = 0
frame_skip = 5  # Process every 5th frame
while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to read frame from video.")
        break

    frame_count += 1

    # Skip frames to speed up processing
    if frame_count % frame_skip != 0:
        continue

    # Predict using Roboflow model
    predictions = model.predict(frame, confidence=50, overlap=50).json()

    # Save detected objects with bounding boxes
    save_detected_objects(frame, predictions, frame_count)

    # Save video
    out.write(frame)

    # Resize frame for display
    resized_frame = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # Optionally display the resized frame
    cv.imshow('Detection Results', resized_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release resources
out.release()
video.release()
cv.destroyAllWindows()

# Save detections to a file
with open('./outputs/detections.txt', 'w') as f:
    for detection in saved_detections:
        f.write(f"{detection}\n")

print("Video processing complete. Detections saved to 'outputs/detections.txt'")
