import torch
import cv2

print("🔁 Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
print("✅ YOLOv5 model loaded.")

# Load video
video_path = "sample_video.mp4"
print(f"🎥 Trying to open video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

print("✅ Video opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⏹️ End of video or can't read frame.")
        break

    print("🎯 Running detection on frame...")
    results = model(frame)
    annotated = results.render()[0]

    cv2.imshow("YOLOv5 Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done.")


