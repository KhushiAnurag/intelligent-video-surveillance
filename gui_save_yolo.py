import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
from PIL import Image, ImageTk
import threading

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Global flags
cap = None
out = None
running = False

# Start detection and saving
def start_detection(video_path):
    global cap, out, running

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video file.")
        return

    # Setup for saving
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_gui.avi", fourcc, fps, (width, height))

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results.render()[0]
        out.write(annotated)  # Save frame

        # Show in GUI
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        window.update_idletasks()
        window.update()

    cap.release()
    out.release()
    messagebox.showinfo("Done", "Detection completed and video saved as 'output_gui.avi'.")

# Stop detection
def stop_detection():
    global running
    running = False

# File selector
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        threading.Thread(target=start_detection, args=(file_path,), daemon=True).start()

# GUI
window = tk.Tk()
window.title("YOLOv5 GUI with Save")
window.geometry("800x600")

video_label = tk.Label(window)
video_label.pack()

btn_frame = tk.Frame(window)
btn_frame.pack(pady=10)

browse_btn = tk.Button(btn_frame, text="Select Video & Start", command=browse_file)
browse_btn.grid(row=0, column=0, padx=10)

stop_btn = tk.Button(btn_frame, text="Stop", command=stop_detection)
stop_btn.grid(row=0, column=1, padx=10)

window.mainloop()
