import cv2
import torch

# Load the YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', source='local')  # use the small model

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Render results on the frame
    rendered_frame = results.render()[0]

    # Display it
    cv2.imshow("Real-Time YOLOv5 Detection", rendered_frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
