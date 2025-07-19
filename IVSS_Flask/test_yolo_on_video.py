import cv2
import torch

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Load your video
cap = cv2.VideoCapture('people_video.mp4')

# Get video properties safely
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

# Define writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_test.mp4', fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Filter for only 'person' class (class id 0)
    results.pred[0] = results.pred[0][results.pred[0][:, 5] == 0]

    # Annotate and write
    annotated = results.render()[0]
    annotated_resized = cv2.resize(annotated, (width, height))
    out.write(annotated_resized)

    frame_count += 1
    print(f"Processed frame {frame_count}")

cap.release()
out.release()
print("✅ Detection done! Check 'output_test.mp4'")
