import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

cap = cv2.VideoCapture("uploads/people_video.mp4")
out = cv2.VideoWriter("static/test_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame)
    annotated = result.render()[0]
    out.write(annotated)

cap.release()
out.release()
