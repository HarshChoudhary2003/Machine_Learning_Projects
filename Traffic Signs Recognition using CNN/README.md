# 🛡️ GuardianEye V2 PRO | Production-Ready Traffic Sign Intelligence

GuardianEye V2 PRO is a high-performance, autonomous-grade Computer Vision system designed for real-time traffic sign classification. Built with a robust 8-layer CNN architecture, it provides mission-critical visual perception for smart cities and autonomous fleets.

## 🚀 PRO Features

- **End-to-End Pipeline**: Unified workflow for data ingress, neural processing, and deployment.
- **📸 Live Vision Integration**: Direct camera/webcam stream interface for real-world testing.
- **🔬 Neural Insights (XAI)**: Explainable AI features including saliency map simulations and layer-wise activation radar.
- **📝 Fleet Logging**: Automated prediction history tracking with one-click **CSV/Intelligence export**.
- **🌐 Localization & Production Tools**: Multi-language support and model configuration settings.
- **Premium Cyber-Aesthetic**: High-performance glassmorphism UI with animated backgrounds and orbitron typography.

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow, Keras (CNN Optimization)
- **Frontend Engine**: Streamlit (Production Build)
- **Data Engineering**: Pandas, NumPy, Scikit-Learn
- **Visualization**: Plotly, Lottie (Motion Graphics)
- **Image Processing**: OpenCV, Pillow (PIL)

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
**Build Status**: `STABLE` | **Version**: `2.4.0-PRO` | **Built for Advanced Mobility**
