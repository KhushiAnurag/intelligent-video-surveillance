import cv2
import torch
import os

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
cap = cv2.VideoCapture('people_video.mp4')

os.makedirs('debug_frames', exist_ok=True)

frame_idx = 0
save_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Print predictions
    print(f"Frame {frame_idx} predictions:", results.pred[0])

    # Keep only persons
    person_preds = results.pred[0][results.pred[0][:, 5] == 0]
    if len(person_preds) == 0:
        print("❌ No person detected.")
    else:
        print(f"✅ {len(person_preds)} person(s) detected.")

    results.pred[0] = person_preds
    annotated = results.render()[0]

    # Save first few frames to debug
    if save_count < 5:
        cv2.imwrite(f'debug_frames/frame_{frame_idx}.jpg', annotated)
        save_count += 1

    frame_idx += 1

cap.release()
print("✅ Done debugging. Check debug_frames/ folder.")
