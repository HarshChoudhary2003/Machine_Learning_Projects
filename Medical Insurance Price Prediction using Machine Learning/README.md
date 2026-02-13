
# 🏥 Medical Insurance Price Prediction

<div align="center">
  <img src="https://img.freepik.com/free-vector/health-insurance-concept-illustration_114360-148.jpg" alt="Medical Insurance AI" width="600" />
</div>

## 📌 Overview
**InsurAI** is an advanced AI-powered application designed to predict medical insurance premiums based on individual health profiles. Leveraging a robust **XGBoost** regression model, this tool provides accurate cost estimations to help users plan their finances better.

The application features a cutting-edge **Streamlit** interface with glassmorphism design, animated elements, and interactive visualizations.

## ✨ Key Features
- **🤖 XGBoost Powered:** Utilizes a high-performance gradient boosting model for precise predictions.
- **🎨 Glassmorphism UI:** A modern, aesthetically pleasing interface with frosted glass effects.
- **🛡️ Lottie Animations:** High-quality vector animations for a dynamic user experience.
- **📊 Interactive Analytics:** 
  - **BMI Gauge:** Visual representation of health status based on BMI.
  - **Risk Factor Breakdown:** Understand how Age, BMI, and Smoking contribute to the cost.
- **📱 Responsive Layout:** Optimized for a seamless experience on various screen sizes.

## 🛠️ Technologies Used
- **Python** (Core Language)
- **Streamlit** (Web Framework)
- **XGBoost** (Machine Learning Model)
- **Pandas & NumPy** (Data Manipulation)
- **Plotly** (Interactive Visualizations)
- **Streamlit-Lottie** (Animations)

## 📂 Project Structure
```
├── app.py                 # Main Streamlit application
├── insurancemodelf.pkl    # Trained XGBoost model
├── insurance.csv          # Dataset used for training
├── main.ipynb             # Jupyter Notebook for EDA and Model Training
└── README.md              # Project Documentation
```

## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed. You can install the required dependencies using:

```bash
pip install streamlit pandas xgboost plotly streamlit-lottie scikit-learn
```

### Running the App
Navigate to the project directory and run:

```bash
streamlit run app.py
```

## 📊 Model Performance
The XGBoost regressor was trained on a dataset containing age, sex, bmi, children, smoker, and region. It achieved a high R² score indicating strong predictive capability on the test set.

## 🤝 Contribution
Contributions are welcome! Feel free to open an issue or submit a pull request.

---
<p align="center">
  Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a>
</p>
