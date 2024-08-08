import cv2 as cv
import os
from roboflow import Roboflow
import easyocr

# Initialize Roboflow
rf = Roboflow(api_key="MMkdGQyPw2nS3ttNfaoZ")
project = rf.workspace().project("platedetector-zkpmk")
model = project.version("11").model

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Directory containing input images
input_image_dir = 'inputs/images'
output_image_dir = 'outputs/img'
output_txt_file = 'outputs/detections.txt'

# Create output directory if it does not exist
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

# Set to keep track of saved detections
saved_detections = set()

# Function to perform OCR on the cropped object
def perform_ocr(cropped_img):
    result = reader.readtext(cropped_img)
    text = ' '.join([item[1] for item in result])
    return text

# Function to save detected objects with bounding boxes
def save_detected_objects(frame, predictions, image_name):
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
                obj_path = os.path.join(output_image_dir, f'{image_name}_{class_name}_{confidence:.2f}.jpg')
                cv.imwrite(obj_path, detected_obj)
                
                # Perform OCR on the cropped object
                text = perform_ocr(detected_obj)
                with open(output_txt_file, 'a') as f:
                    f.write(f"{obj_path}: {text}\n")
                
                saved_detections.add(detection_id)
                print(f"Saved detection: {detection_id}")

# Process each image in the directory
for image_name in os.listdir(input_image_dir):
    image_path = os.path.join(input_image_dir, image_name)
    
    # Check if the file is an image
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read the image
        frame = cv.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image file {image_path}.")
            continue
        
        # Predict using Roboflow model
        predictions = model.predict(frame, confidence=50, overlap=50).json()

        # Save detected objects with bounding boxes
        save_detected_objects(frame, predictions, image_name)
        
        # Optionally display the frame
        cv.imshow('Detection Results', cv.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))))
        if cv.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to exit
            break

# Finalize
cv.destroyAllWindows()

print("Image processing complete. Detections and OCR results saved.")
