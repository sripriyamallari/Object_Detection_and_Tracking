# 🎯 Object Detection and Tracking

A Computer Vision project developed as part of my **CodeAlpha Internship**.


This project uses **YOLO** for object detection and **ByteTrack** for multi-object tracking. It detects objects in uploaded videos, assigns unique tracking IDs, draws bounding boxes, and displays the processed video through a Streamlit web application.


## 🚀 Live Demo

🔗 https://objectdetectionandtracking-opb2yqzgxzfrbrkvwfreog.streamlit.app/


## ✨ Features

- 🎯 Object detection using YOLO
- 🔄 Multi-object tracking using ByteTrack
- 🆔 Unique tracking IDs
- 📦 Bounding boxes around detected objects
- 🎥 Video upload and processing
- 🌐 Interactive Streamlit web application
- 📥 Download processed tracking video


## 🛠️ Technologies Used

- Python
- YOLO
- ByteTrack
- OpenCV
- Streamlit
- NumPy


## ⚙️ How It Works

1. Upload a video to the Streamlit application.
2. YOLO detects objects in each video frame.
3. ByteTrack tracks detected objects across frames.
4. Each tracked object receives a unique ID.
5. Bounding boxes and tracking IDs are displayed.
6. The processed video is generated and can be downloaded.


## 📂 Project Structure

```text
Object_Detection_and_Tracking/
│
├── app.py
├── Object_Detection_and_Tracking.ipynb
├── requirements.txt
├── packages.txt
├── README.md
└── .gitignore