import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_and_clean_data, feature_engineering, prepare_models_data

# Ensure paths exist
BASE_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'notebooks'), exist_ok=True) # For saving some plots

def run_pipeline():
    # 1. Load and Cleaning
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    df = load_and_clean_data(data_path)
    print("✅ Data Loaded and Cleaned")

    # 2. Feature Engineering
    df = feature_engineering(df)
    print("✅ Feature Engineering Completed")

    # 3. EDA - Basic Visualizations
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='continent_name')
    plt.title('User Distribution across Continents')
    plt.savefig(os.path.join(BASE_DIR, 'notebooks/continent_dist.png'))
    plt.close()

    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='attraction_type_name', y='rating')
    plt.title('Rating Distribution across Attraction Types')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(BASE_DIR, 'notebooks/rating_dist.png'))
    plt.close()

    plt.figure(figsize=(10,8))
    sns.heatmap(df[['rating', 'visit_year', 'visit_month', 'user_age', 'avg_user_rating', 'popularity_score']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(BASE_DIR, 'notebooks/correlation.png'))
    plt.close()

    # 4. Model Training - Prepare Data
    X, y_reg, y_clf, feature_names, scaler = prepare_models_data(df)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # 4.1 Regression (Predicting Rating)
    print("\n--- Training Regression Models ---")
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_r, y_train_r)
    y_pred_r = reg_model.predict(X_test_r)
    print(f"Random Forest Regressor R2: {r2_score(y_test_r, y_pred_r):.4f}")
    print(f"Random Forest Regressor RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_r)):.4f}")

    # 4.2 Classification (Predicting Visit Mode)
    print("\n--- Training Classification Models ---")
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_c, y_train_c)
    y_pred_c = clf_model.predict(X_test_c)
    print(f"Random Forest Classifier Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")

    # Save models and scaler
    with open(os.path.join(BASE_DIR, 'models/reg_model.pkl'), 'wb') as f:
        pickle.dump(reg_model, f)
    with open(os.path.join(BASE_DIR, 'models/clf_model.pkl'), 'wb') as f:
        pickle.dump(clf_model, f)
    with open(os.path.join(BASE_DIR, 'models/scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(BASE_DIR, 'models/feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save the cleaned dataframe for recommendation system
    df.to_csv(os.path.join(BASE_DIR, 'data/final_processed_data.csv'), index=False)
    print("✅ Models saved Successfully")

    # 5. Recommendation System (Collaborative Filtering Example)
    print("\n--- Building Recommendation Matrices ---")
    # User-Item matrix
    rating_matrix = df.pivot_table(index='user_id', columns='attraction_name', values='rating').fillna(0)
    user_similarity = cosine_similarity(rating_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)
    
    with open(os.path.join(BASE_DIR, 'models/user_similarity.pkl'), 'wb') as f:
        pickle.dump(user_similarity_df, f)
    with open(os.path.join(BASE_DIR, 'models/rating_matrix.pkl'), 'wb') as f:
        pickle.dump(rating_matrix, f)
    print("✅ Recommendation artifacts saved")

if __name__ == "__main__":
    run_pipeline()
