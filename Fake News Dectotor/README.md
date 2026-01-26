# 📰 Fake News Detection and Data Explorer Dashboard

## Project Overview

This project is a Python-based interactive web application built using **Streamlit**. It serves two main purposes:

1.  **Data Exploration:** It provides a detailed visual dashboard to explore the two provided datasets (`Fake.csv` and `True.csv`), comparing key features like article subject distribution and text length.
2.  **Real-time Prediction:** It allows users to paste a new news article into the sidebar and instantly runs a basic **Machine Learning (ML) classifier** to predict whether the article is likely **Fake** or **Real** news.

This application demonstrates a complete data science workflow, from data loading and visualization to model training and deployment.

## ✨ Features

* **Interactive Dashboard:** Visualize article counts, unique subjects, and text length distributions using dynamic Plotly charts.
* **Real-time Analysis:** Dedicated sidebar form for pasting new articles.
* **Instant Prediction:** The integrated Logistic Regression model provides an immediate classification (Fake/Real) and a confidence score for user-submitted text.
* **Cached Performance:** The ML model is trained once and cached (`@st.cache_resource`), ensuring fast predictions and application responsiveness.

## 🛠️ Technology Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Data Analysis:** Pandas
* **Visualization:** Plotly Express
* **Machine Learning:** Scikit-learn (for TF-IDF Vectorization and Logistic Regression)

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

You need Python (3.7+) installed on your system.

1.  **Clone the Repository (or Download Files):**

    ```bash
    git clone [https://github.com/HarshChoudhary2003/fake-news-dashboard.git]
    cd fake-news-dashboard
    ```
    *(Replace the URL with your actual GitHub repository URL)*

2.  **Install Dependencies:**
    Install all required Python packages using `pip`.

    ```bash
    pip install streamlit pandas plotly scikit-learn numpy
    ```

3.  **Ensure Data Files are Present:**
    Make sure the following files are in your project directory:
    * `creative_app.py` (The main application file)
    * `Fake.csv`
    * `True.csv`

### Running the Application

1.  **Run the Streamlit App:**
    Execute the main Python file using the Streamlit CLI.

    ```bash
    streamlit run creative_app.py
    ```

2.  **Access the App:**
    Your browser should automatically open the application at `http://localhost:8501`.

## 🧠 Model Details

The fake news classification relies on a **text processing pipeline** trained on the combined `Fake.csv` and `True.csv` datasets:

* **Vectorizer:** **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert the article text into a numerical format, capturing the importance of words.
* **Classifier:** A **Logistic Regression** model is trained on the vectorized text to classify the article content.
* **Training:** The model training occurs automatically on the first launch of the `creative_app.py` file and is cached, so it does not re-train on every refresh.

## ✍️ Usage Instructions

1.  **Data Exploration:** Use the default view (**Summary Dashboard**) and the tabs provided to explore the existing data distributions.
2.  **Article Prediction:**
    * Go to the **sidebar**.
    * In the **"Analyze New Article"** section, paste the **Article Title** and the **Article Text**.
    * Click the **"Analyze Article"** button.
    * The main view will switch to **"New Article Initial Analysis and Prediction"**, showing the text features and the model's classification (REAL/FAKE) and confidence score.

---
