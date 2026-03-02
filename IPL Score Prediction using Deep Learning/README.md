<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=0,20,40&height=250&section=header&text=IPL%20Score%20AI&fontSize=40&animation=fadeIn&fontAlignY=38&desc=Deep%20Learning%20Sports%20Intelligence&descAlignY=55&descAlign=50" width="100%" />

  <br/>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
  [![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)](https://github.com/HarshChoudhary2003)

</div>

---

## 🏏 Overview
An advanced **AI-powered cricket intelligence platform** that predicts final projected scores for IPL matches. Utilizing a high-accuracy Deep Learning Neural Network (`MLPRegressor`), the system analyzes real-time momentum, ground conditions, and historical trends.

### 🌟 Project Armor
- 🧠 **Neural Core**: Multi-layer Perceptron (MLP) architecture.
- ⚡ **Momentum Engine**: Features tracking CRR, Balls Left, and Wicket impact.
- 🎨 **Hyper-Modern UI**: Glassmorphism dashboard with neon tactical accents.
- 📊 **Live Analytics**: Real-time score projection updates.

---

## 🏗️ Neural Architecture
```mermaid
graph TD
    Input[Match telemetry] --> Processing[One-Hot Encoding]
    Processing --> Scaler[StandardScaler]
    Scaler --> MLP[Deep Neural Network]
    MLP --> Output[Projected Final Score]
    style MLP fill:#293742,stroke:#fff,stroke-width:2px,color:#fff
```

---

## 🚀 Deployment Manual
1. **Core Ignition**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch Visualizer**:
   ```bash
   streamlit run app.py
   ```

---

<div align="center">
  <h3>⭐ If you found this sports intelligence core useful, please consider giving it a star!</h3>
  <p>Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a></p>
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=0,20,40&height=100&section=footer" width="100%" />
</div>
