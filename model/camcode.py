import cv2
import torch
from ultralytics import YOLO #Import Yolo Library

# Load your trained YOLOv8 model
model = YOLO('best.pt')

# Start video capture from the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam, in case of using a phone camera with IP Webcam replace the "0" with your URL HTTP://<IP>:<Port>/Video 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize drowsy detection counter and threshold
# start counting from moment it detects a drowsy driver before alerting him
drowsy_counter = 0
warning_threshold = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Perform YOLOv8 inference on the frame (maintaining original size for accuracy)
    results = model(frame)

    drowsy_detected = False  # Flag to check if drowsy state is detected in current frame

    # Process YOLOv8 results and draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates in (x1, y1, x2, y2) format
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            c = int(box.cls[0])
            conf = box.conf[0]
            label = f'{model.names[c]} {conf:.2f}'
            color = (0, 255, 0)  # Green color for bounding box

            # Check if the detected class is "drowsy" (adjust as needed)
            if model.names[c] == 'drowsy':
                drowsy_detected = True
                color = (0, 0, 255)  # Red color for drowsy detection

            # Draw bounding box around detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Put label above the bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Update drowsy counter
    if drowsy_detected:
        drowsy_counter += 1
    else:
        drowsy_counter = 0

    # Display warning if drowsy counter exceeds threshold
    if drowsy_counter > warning_threshold:
        warning_text = "WARNING: Drowsiness Detected!"
        cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame with bounding boxes
    cv2.imshow('YOLOv8 Real-Time Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
