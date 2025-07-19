import cv2
import torch

# ✅ Load YOLOv5s from GitHub (no local files required!)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# ✅ Load video file
cap = cv2.VideoCapture("sample_video.mp4")  # Replace with your video filename

if not cap.isOpened():
    print("❌ Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video finished.")
        break

    results = model(frame)
    annotated = results.render()[0]

    cv2.imshow("YOLOv5 Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

