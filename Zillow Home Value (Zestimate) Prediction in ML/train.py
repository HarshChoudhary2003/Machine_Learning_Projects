import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
try:
    from xgboost import XGBRegressor
    MODEL_TYPE = "XGB"
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    MODEL_TYPE = "RF"
from sklearn import metrics
import joblib
import os

# Set Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Zillow.csv")
MODEL_DIR = os.path.join(BASE_DIR, "zillow_app", "model")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

print("Cleaning data...")
# Replicating notebook logic
to_remove = []
for col in df.columns:
    if df[col].nunique() == 1:
        to_remove.append(col)
    elif (df[col].isnull()).mean() > 0.60:
        to_remove.append(col)

df.drop(to_remove, axis=1, inplace=True)

# Remove outliers based on notebook
df = df[(df['target'] > -1) & (df['target'] < 1)]

# Impute missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    elif df[col].dtype == np.number:
        df[col] = df[col].fillna(df[col].mean())

# Encode Categorical Variables
encoders = {}
for col in df.columns:
    if col == 'target':
        continue
    
    # Check if column is not numeric
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Encoding non-numeric column: {col}")
        # Convert to string first to handle mixed types safely
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Save metadata about columns for the UI
feature_columns = [col for col in df.columns if col != 'target']
meta_data = {
    'features': feature_columns,
    'encoders': encoders, # Note: LabelEncoders might be large to pickle directly if highly unique, but for this dataset it should be fine
    'feature_stats': df[feature_columns].describe().to_dict() # Useful for min/max in UI
}

print("Training Model...")
X = df.drop(['target'], axis=1)
Y = df['target']

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=1)

if MODEL_TYPE == "XGB":
    print("Using XGBRegressor...")
    model = XGBRegressor()
else:
    print("XGBoost not found, using RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=10, max_depth=10) # reduced for speed
    
model.fit(X_train, Y_train)

print("Evaluating Model...")
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

print('Training MAE : ', metrics.mean_absolute_error(Y_train, train_preds))
print('Validation MAE : ', metrics.mean_absolute_error(Y_val, val_preds))

print("Saving Artifacts...")
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(meta_data, os.path.join(MODEL_DIR, "meta_data.pkl"))

print("Done!")
