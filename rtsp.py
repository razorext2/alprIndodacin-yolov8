import cv2 as cv
from ultralytics import YOLO
import os

# RTSP URL for the camera
rtsp_url = 'rtsp://admin:admin123@192.168.11.34:554/sub_stream'

# Initialize the pre-trained YOLO model
model_pretrained = YOLO('yolov8n.pt')  # No weights_only argument

# Open a connection to the RTSP stream
video = cv.VideoCapture(rtsp_url)

# Check if the video stream opened successfully
if not video.isOpened():
    print(f"Error: Could not open RTSP stream {rtsp_url}.")
    exit()

# Get video dimensions
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

# Calculate the new dimensions for the output window (50% of original size)
new_width = int(frame_width * 0.5)
new_height = int(frame_height * 0.5)
new_size = (new_width, new_height)

# Define the codec and create VideoWriter object with new size
fourcc = cv.VideoWriter_fourcc(*'DIVX')
output_path = './outputs/output.mp4'
if not os.path.exists('./outputs'):
    os.makedirs('./outputs')
out = cv.VideoWriter(output_path, fourcc, 10.0, new_size)  # You can adjust FPS here if needed

# Read and process frames
while True:
    ret, frame = video.read()
    if ret:
        try:
            # Perform inference on the frame
            results = model_pretrained(frame)

            # Check if results contain detected objects
            if results:
                # Plot results on the frame
                composed = results[0].plot()

                # Resize frame to new size for display and saving
                resized_frame = cv.resize(composed, new_size)

                # Save video
                out.write(resized_frame)

                # Optionally display the resized frame
                cv.imshow('Detection Results', resized_frame)
                if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    break
            else:
                print("No results from detection.")
        except Exception as e:
            print(f"Error during detection: {e}")
    else:
        print("Error: Failed to read frame from video.")
        break

# Release resources
out.release()
video.release()
cv.destroyAllWindows()