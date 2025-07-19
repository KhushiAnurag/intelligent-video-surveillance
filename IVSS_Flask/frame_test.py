import cv2

cap = cv2.VideoCapture("people_video.avi")

ret, frame = cap.read()
if not ret:
    print("❌ Still not working, bro 😭")
else:
    print("✅ Frame read successfully!")
    cv2.imwrite("avi_test_frame.jpg", frame)

cap.release()

