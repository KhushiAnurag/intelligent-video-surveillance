import os
from flask import Flask, render_template, request, send_from_directory
import cv2
import torch
import sys
import numpy as np

# Add local yolov5 folder to system path
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

# Import YOLOv5 components
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
device = select_device('')
model = attempt_load('yolov5s.pt', map_location=device)
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    download_link = None
    if request.method == 'POST':
        video = request.files['video']
        if video.filename.endswith('.mp4'):
            input_path = os.path.join(UPLOAD_FOLDER, video.filename)
            video.save(input_path)

            output_path = os.path.join(OUTPUT_FOLDER, 'output.mp4')
            detect_persons(input_path, output_path)
            download_link = '/static/output.mp4'

    return render_template('index.html', download_link=download_link)

def detect_persons(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = letterbox(frame, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) == 0:  # class 0 = person
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                      (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

@app.route('/static/<filename>')
def serve_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)




