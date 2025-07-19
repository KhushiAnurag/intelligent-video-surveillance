import cv2
import torch
import os

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Create output folder
os.makedirs('frames_manual', exist_ok=True)

# Load video
cap = cv2.VideoCapture('people_video.mp4')
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.pred[0]

    # Filter person class (class ID 0)
    detections = detections[detections[:, 5] == 0]

    # Draw manually
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        label = f'Person {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save debug frame
    if frame_idx < 5:
        cv2.imwrite(f'frames_manual/frame_{frame_idx}.jpg', frame)

    frame_idx += 1
    print(f"✅ Processed frame {frame_idx}")

cap.release()
print("🎉 Done! Check frames_manual/ folder for output.")
