import cv2

cap = cv2.VideoCapture('people_video.mp4')
if not cap.isOpened():
    print("❌ Could not open the video file!")
else:
    print("✅ Video file opened successfully!")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No more frames to read or can't read frame.")
        break

    frame_count += 1
    print(f"Read frame {frame_count}")

cap.release()
print(f"📸 Total frames read: {frame_count}")
