<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=0,20,40&height=250&section=header&text=GuardianEye%20V2%20PRO&fontSize=40&animation=fadeIn&fontAlignY=38&desc=Production-Ready%20Traffic%20Sign%20Intelligence&descAlignY=55&descAlign=50" width="100%" />

  <br/>

  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
  [![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
  [![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [PRO Features](#-pro-features)
- [Technology Stack](#️-technology-stack)
- [Production Deployment](#-production-deployment)
- [Neural Architecture](#-neural-architecture-blueprint)

---

## 🛡️ Overview

**GuardianEye V2 PRO** is a high-performance, autonomous-grade Computer Vision system designed for real-time traffic sign classification. Built with a robust **8-layer CNN architecture**, it provides mission-critical visual perception for smart cities and autonomous fleets.

## 🚀 PRO Features

- **✅ End-to-End Pipeline**: Unified workflow for data ingress, neural processing, and deployment.
- **📸 Live Vision Integration**: Direct camera/webcam stream interface for real-world testing.
- **🔬 Neural Insights (XAI)**: Explainable AI features including saliency map simulations and layer-wise activation radar.
- **📝 Fleet Logging**: Automated prediction history tracking with one-click **CSV/Intelligence export**.
- **🌐 Localization & Production Tools**: Multi-language support and model configuration settings.
- **✨ Premium Cyber-Aesthetic**: High-performance glassmorphism UI with animated backgrounds and Orbitron typography.

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | TensorFlow, Keras (CNN Optimization) |
| **Frontend Engine** | Streamlit (Production Build) |
| **Data Engineering** | Pandas, NumPy, Scikit-Learn |
| **Visualization** | Plotly, Lottie (Motion Graphics) |
| **Image Processing** | OpenCV, Pillow (PIL) |

---

## 📦 Production Deployment

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Synchronization**:
   Ensure `traffic_sign_model.h5` is present in the root. If not, run the training engine:
   ```bash
   python train.py
   ```

3. **System Launch**:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Neural Architecture Blueprint

```python
# CNN Stack: 8 Specialized Layers
- Conv2D (32 filters, 5x5) -> Spatial feature extraction
- Conv2D (32 filters, 5x5) -> Hierarchical pattern recognition
- MaxPool2D (2x2) -> Dimensionality reduction
- Conv2D (64 filters, 3x3) -> High-level structural detection
- Conv2D (64 filters, 3x3) -> Class-specific feature mapping
- MaxPool2D (2x2) -> Translation invariance
- Dense (256 units) -> Global feature integration
- Softmax (43 units) -> Probability distribution
```

---

<div align="center">
  <p><b>Build Status</b>: <code>STABLE</code> | <b>Version</b>: <code>2.4.0-PRO</code> | <b>Built for Advanced Mobility</b></p>
  
  <h3>⭐ If you found GuardianEye useful, please consider giving it a star!</h3>
  <p>Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a></p>
  
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=0,20,40&height=100&section=footer" width="100%" />
</div>
