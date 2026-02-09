# 👤 Real-Time Face Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-6804D6?style=for-the-badge&logo=roboflow&logoColor=white)
![Supervision](https://img.shields.io/badge/Supervision-FF6F00?style=for-the-badge&logo=opencv&logoColor=white)

A powerful real-time face recognition and tracking system leveraging the **Roboflow API** and **Supervision** library. This project detects faces in video streams, tracks them across frames, and provides annotated video output.

## 🌟 Key Features

- **Real-Time Detection**: Utilizes Roboflow's inference API for accurate face detection.
- **Advanced Tracking**: Implements **ByteTrack** for robust object tracking across frames.
- **Zone Monitoring**: Defines specific regions of interest (PolygonZone) for focused analysis.
- **Video Annotation**: Generates high-quality annotated videos with bounding boxes and labels.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **Roboflow**: For accessing the hosted face recognition model.
- **Supervision**: For video processing, annotation, and tracking utilities.
- **NumPy**: For numerical operations and array handling.

## 📋 Prerequisites

Ensure you have Python installed. You also need a Roboflow API key.

```bash
pip install roboflow supervision numpy
```

## 🚀 Usage

### 1. Single Image Inference

Run `face recognition.py` to detect faces in a single image:

```bash
python "face recognition.py"
```
This script sends an image to the Roboflow API and saves the visual prediction.

### 2. Video Processing & Tracking

Run `count face.py` or `opencv.py` to process a video file (`sample 1.mp4`):

```bash
python "count face.py"
```

This script:
1. Loads the video.
2. Detects faces in each frame using the Roboflow model.
3. Tracks detections using ByteTrack.
4. Annotates the frames with bounding boxes and labels.
5. Saves the output to `video_out.mp4` or `detect.mp4`.

## 📂 Project Structure

- `face recognition.py`: Script for single image inference.
- `count face.py`: Main script for video processing and counting.
- `opencv.py`: Alternative script for video detection.
- `sample 1.mp4`: Input video file (ensure this exists).

## 📝 Configuration

You need to set your Roboflow API Key in the scripts:

```python
rf = Roboflow(api_key="YOUR_API_KEY")
```

*Note: The current code references a specific project `real-time-facial-recognition` version 1.*
