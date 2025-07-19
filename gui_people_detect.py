import tkinter as tk
from tkinter import messagebox
import cv2
import torch
from PIL import Image, ImageTk

# Load YOLOv5s model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Globals
cap = None
out = None
running = False

def process_frame():
    global cap, out, running

    if running and cap:
        ret, frame = cap.read()
        if ret:
            # YOLO detection
            results = model(frame)

            # Only detect persons (class ID 0)
            person_detections = results.pred[0][results.pred[0][:, 5] == 0]
            results.pred[0] = person_detections
            annotated = results.render()[0]

            out.write(annotated)

            # Convert BGR → RGB → Image → ImageTk
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        else:
            stop_detection()
            messagebox.showinfo("Done", "Person detection complete. Saved as 'output_gui.avi'")
            return

        # Schedule next frame
        window.after(30, process_frame)

def start_detection():
    global cap, out, running

    # Open your saved video file
    cap = cv2.VideoCapture("people_video.mp4")
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open 'people_video.mp4'")
        return

    # Output video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_gui.avi", fourcc, fps, (width, height))

    running = True
    process_frame()

def stop_detection():
    global running, cap, out
    running = False
    if cap:
        cap.release()
    if out:
        out.release()

# Tkinter GUI setup
window = tk.Tk()
window.title("Person Detection - YOLOv5 GUI")
window.geometry("800x600")

video_label = tk.Label(window)
video_label.pack()

btn_frame = tk.Frame(window)
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame, text="Start Detection", command=start_detection)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = tk.Button(btn_frame, text="Stop", command=stop_detection)
stop_btn.grid(row=0, column=1, padx=10)

window.mainloop()
