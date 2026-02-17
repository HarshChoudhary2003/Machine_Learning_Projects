import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ipl_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ipl_model.pkl")

def train_advanced_model():
    print("🚀 Loading Dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    # --- 1. Data Cleaning & Feature Engineering ---
    print("🧹 Cleaning & Preprocessing Data...")
    
    # Consistent Teams
    consistent_teams = [
        'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
        'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
        'Delhi Daredevils', 'Sunrisers Hyderabad'
    ]
    
    # Filter teams and valid overs
    df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
    df = df[df['overs'] >= 5.0]

    # Additional Features
    # Convert 'date' to datetime (optional, maybe check year trend later, skipping for now)
    
    # Feature 1: Current Run Rate (CRR)
    df['crr'] = df['runs'] / df['overs']
    
    # Feature 2: Balls Left
    df['balls_left'] = 120 - (df['overs'] * 6)
    
    # Feature 3: Wickets Left
    df['wickets_left'] = 10 - df['wickets']
    
    # Feature 4: Last 5 Overs Run Rate
    # Assuming 'runs_last_5' is available
    df['last_5_rr'] = df['runs_last_5'] / 5.0

    # Select Features and Target
    # Categorical: 'bat_team', 'bowl_team', 'venue'
    # Numerical: 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'crr', 'balls_left', 'wickets_left'
    
    X = df[['bat_team', 'bowl_team', 'venue', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'crr', 'balls_left', 'wickets_left']]
    y = df['total']

    # --- 2. Advanced Preprocessing Pipeline ---
    print("⚙️ Building Transformation Pipeline...")
    
    # Categorical Pipeline: OneHotEncoding
    categorical_features = ['bat_team', 'bowl_team', 'venue']
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') # Avoid dummy trap
    
    # Numerical Pipeline: Standardization
    numerical_features = ['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'crr', 'balls_left', 'wickets_left']
    numerical_transformer = StandardScaler()
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='passthrough'
    )

    # --- 3. Deep Learning Model Architecture ---
    # MLPRegressor with more layers and regularization
    model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128, 64), # Deeper and wider architecture
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )

    # Create Full Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    # --- 4. Training ---
    print("🧠 Training Advanced Neural Network...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline.fit(X_train, y_train)
    
    # --- 5. Evaluation ---
    print("📊 Evaluating Model Performance...")
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Training Complete!")
    print(f"📉 Mean Absolute Error: {mae:.2f} runs")
    print(f"📈 R-Squared Score: {r2:.4f}")
    
    if r2 > 0.75:
        print("🏆 Excellent Accuracy Achieved!")
    elif r2 > 0.60:
        print("👍 Good Accuracy.")
    else:
        print("⚠️ Accuracy could be better.")

    # --- 6. Save Artifacts ---
    print(f"💾 Saving High-Precision Model to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    print("🎉 All Done! You can now run the app.")

if __name__ == "__main__":
    train_advanced_model()
