# Tourism Experience Analytics 🌍

A strong portfolio-level data science project involving Regression, Classification, and a Recommendation System for the Tourism Industry.

## 📁 Project Structure

```
Tourism_Project/
│
├── data/               # Raw and processed datasets
├── notebooks/          # EDA plots and saved visualizations
├── models/             # Saved ML models (Pickle files)
├── app.py              # Streamlit Dashboard
├── train_models.py     # Training pipeline script
├── utils.py            # Reusable cleaning & feature engineering functions
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## 🚀 Features

### 1️⃣ Regression (Rating Prediction)
Predicts the **Attraction Rating (1-5)** based on user demographics and attraction popularity using **Random Forest Regressor**.

### 2️⃣ Classification (Visit Mode Prediction)
Predicts the **Visit Mode** (Business, Family, Couples, etc.) using a **Random Forest Classifier**.

### 3️⃣ Recommendation System
Suggests top 5 attractions for users using **Collaborative Filtering** (User-Item similarity).

### 4️⃣ Interactive Dashboard
Built with **Streamlit**, allowing users to input their preferences and see real-time predictions and recommendations.

## 🛠️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data & Train Models:** (If you want to re-run from scratch)
   ```bash
   python data/generate_data.py
   python train_models.py
   ```

3. **Run Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## 🧠 Master Strategy Applied
- **Phase 1:** Data Cleaning Pipeline (Handling missing values, outliers, and merging 9+ tables).
- **Phase 2:** Feature Engineering (User/Attraction aggregation, encoding).
- **Phase 3:** Advanced EDA (Visualizing distributions and correlations).
- **Phase 4:** Model Building (Training & Evaluating multiple ML models).
- **Phase 5:** Deployment (Professional Streamlit UI).

Developed to boost the Data Science portfolio for Harsh 🔥
