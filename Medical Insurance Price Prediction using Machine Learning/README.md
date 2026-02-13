
# 🏥 Medical Insurance Price Prediction
  <br>
  
  ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
  ![XGBoost](https://img.shields.io/badge/XGBoost-15FA00?style=for-the-badge&logo=xgboost&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
  ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
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

## 📊 Dataset Details
The model is trained on a medical insurance dataset containing the following features:
- **Age:** Age of the primary beneficiary.
- **Sex:** Insurance contractor gender (female/male).
- **BMI:** Body mass index (kg/m^2).
- **Children:** Number of children covered by health insurance.
- **Smoker:** Smoking status (yes/no).
- **Region:** The beneficiary's residential area in the US (northeast, southeast, southwest, northwest).
- **Charges:** Individual medical costs billed by health insurance.

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
Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Installation
Install the required dependencies:

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

## 📸 Screenshots
*(Add screenshots of your application here)*

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contribution
Contributions are welcome! Feel free to open an issue or submit a pull request.

---
<p align="center">
  Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a>
</p>
