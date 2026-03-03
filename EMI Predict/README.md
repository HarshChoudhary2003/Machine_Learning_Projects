<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=0,40,20&height=250&section=header&text=EMI%20Precision%20AI&fontSize=40&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Financial%20Risk%20Assessment&descAlignY=55&descAlign=50" width="100%" />

  <br/>

  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
  [![XGBoost](https://img.shields.io/badge/XGBoost-28A745?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai)
  [![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
  [![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

</div>

**EMI Precision AI** is a next-generation financial assessment tool designed to evaluate loan eligibility and predict the maximum affordable EMI for users in real-time. 

Unlike traditional calculators that use fixed formulas, this system leverages **Machine Learning (XGBoost)** to analyze a user's comprehensive financial profile—including income, expenses, credit score, and existing liabilities—to provide a personalized risk assessment.

---

## ✨ Key Features

- **🚀 Real-Time AI Assessment**: Instant evaluation of loan eligibility (Eligible / High Risk / Not Eligible).
- **📊 Predictive Analytics**: Estimates the **Maximum Affordable EMI** dynamically based on spending patterns.
- **🎨 Glassmorphism UI**: A stunning, modern dark-mode interface with animated interactions.
- **📈 Interactive Dashboard**: Visualizes credit score trends, approval rates, and financial health.
- **🛠️ MLOps Integration**: Built with **MLflow** for robust model lifecycle management.

---

## 📸 Interface Preview

*(Add screenshots of the Dashboard and Predictor here)*

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Lottie Animations, Plotly
- **Machine Learning**: XGBoost (Gradient Boosting)
- **Model Orchestration**: MLflow
- **Data Processing**: Pandas, NumPy, Scikit-Learn

---

## 🚀 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Machine_Learning_Projects/EMI Predict"
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   The app will open in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

1. **User Input**: Users provide financial details (Salary, Rent, Expenses, Credit Score, etc.).
2. **Feature Engineering**: The system calculates critical metrics like **Debt-to-Income (DTI) Ratio** and **Affordability Index**.
3. **ML Inference**: 
   - A **Classifier** determines eligibility status.
   - A **Regressor** predicts the safe EMI limit.
4. **Visual Output**: Results are displayed with explainable insights and interactive charts.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

<div align="center">
  <h3>⭐ If you found EMI Precision AI useful, please consider giving it a star!</h3>
  <p>Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a></p>
  
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=0,40,20&height=100&section=footer" width="100%" />
</div>
