# Intelligent Video Surveillance System (IVSS)

A real-time person detection web app using **YOLOv5**, **Flask**, and **OpenCV** — deployed on **Render**.

![IVSS Demo Banner](https://your-banner-link.com/demo.gif)

---

## 🔗 Live Demo

**[Access the Web App →](https://intelligent-video-surveillance-1.onrender.com/)**

---

## 📌 Features

* Upload `.mp4` videos via the web interface
* Detects only **persons** using YOLOv5
* Highlights detected persons with bounding boxes
* Downloads the processed video with annotations
* Responsive, animated, and modern dark-themed frontend

---

## 📁 Project Structure

```
intelligent-video-surveillance/
├── app.py                # Flask backend
├── yolov5/               # YOLOv5 local repository
├── yolov5s.pt            # Pre-trained model weights
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html        # Frontend HTML (Tailwind + Jinja)
├── static/              # Output video gets saved here
├── uploads/             # Uploaded video files
└── README.md
```

---

## 🧠 How It Works

1. User uploads a `.mp4` file
2. Flask saves the file in `uploads/`
3. YOLOv5 detects **only persons** from video frames
4. Annotated output is saved in `static/output.mp4`
5. User gets a download link and live video preview

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repository
$ git clone https://github.com/KhushiAnurag/intelligent-video-surveillance.git
$ cd intelligent-video-surveillance

# 2. Create a virtual environment (optional but recommended)
$ python -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Run the Flask app
$ python app.py

# Open in browser
http://localhost:5000
```

> Ensure `yolov5s.pt` is present in the root and `yolov5/` is properly cloned.

---

## 🚀 Deployment (Render)

* Link your GitHub repo to [Render](https://render.com)
* Add Build Command:

```bash
pip install -r requirements.txt
```

* Start Command:

```bash
gunicorn app:app
```

* Select Python 3.10+ and deploy.

> **Important**: Use local YOLOv5 (not torch.hub) to avoid rate-limiting.

---

## 🧰 Tech Stack

* **Frontend**: HTML, Tailwind CSS, JS, Jinja2
* **Backend**: Flask
* **Model**: YOLOv5 (Local)
* **Processing**: OpenCV
* **Deployment**: Render

---

## 🧩 Future Improvements

* Live camera detection (WebCam support)
* Multiple object class selection
* Real-time alert system
* Email/Telegram alerts

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Khushi Anurag**
\:octocat: [GitHub](https://github.com/KhushiAnurag)  •  💼 [LinkedIn](https://www.linkedin.com/in/khushi-anurag)

---

> If you found this project helpful, don't forget to ⭐ star the repository!
Added project README with full instructions and demo link
