import cv2
import torch
import imageio_ffmpeg

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Load your test video
cap = cv2.VideoCapture("people_video.mp4")

# Extract video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

# Set up writer
writer = imageio_ffmpeg.write_frames("final_output.mp4", size=(width, height), fps=fps)
writer.send(None)  # start generator manually

# Process and write frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    results.pred[0] = results.pred[0][results.pred[0][:, 5] == 0]  # person only
    annotated = results.render()[0]

    # Convert BGR to RGB for FFmpeg
    rgb_frame = annotated[:, :, ::-1]
    writer.send(rgb_frame)

    frame_count += 1
    print(f"✅ Processed frame {frame_count}")

# Release resources
cap.release()
writer.close()
print("🎉 final_output.mp4 written successfully and should be playable now.")
