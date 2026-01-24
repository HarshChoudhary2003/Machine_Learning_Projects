# EMI Prediction AI 💳

A machine learning project designed to predict Equated Monthly Installment (EMI) outcomes and risk factors using banking and loan datasets.

## Project Structure
- `app/streamlit_app.py`: Interactive web interface for EMI prediction.
- `notebooks/EDA.ipynb`: Exploratory Data Analysis and visualization.
- `src/train_regression.py`: Training pipeline for the regression model.
- `data/`: Contains processed and raw datasets (in parquet format).

## Features
- MLflow integration for experiment tracking.
- Preprocessed data pipelines.
- Streamlit-based deployment for real-time inference.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
