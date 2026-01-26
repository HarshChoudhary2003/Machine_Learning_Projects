import pandas as pd
import numpy as np
import sqlite3
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- 1. LOAD DATA ---
print("Loading data from DB...")
conn = sqlite3.connect('jobs.db')
df = pd.read_sql("SELECT * FROM jobs_cleaned", conn)
conn.close()

# --- 2. DATA PROGRAMMING (Creating the Target Variable) ---
# Since we don't have real salary, we estimate it based on Indian Market Standards
# This is called "Weak Supervision" - using rules to generate labels.

def estimate_salary(row):
    # Base Salary: 3 LPA + (Experience * 2.5 LPA)
    base = 3.0 + (row['Min_Exp'] * 2.5)
    
    # Skill Bonuses
    skills = str(row['Skills_Detected']).lower()
    if 'python' in skills: base += 1.5
    if 'machine learning' in skills: base += 2.0
    if 'aws' in skills: base += 1.5
    
    # Location Adjustments
    loc = str(row['Location_Clean']).lower()
    if 'bangalore' in loc or 'bengaluru' in loc: base += 1.0
    if 'mumbai' in loc: base += 1.0
    
    # Random noise to make it realistic (Machine Learning needs variation)
    noise = np.random.uniform(-1.0, 1.0)
    
    return round(base + noise, 2)

print("Generating Salary Labels...")
df['Estimated_Salary'] = df.apply(estimate_salary, axis=1)

# --- 3. FEATURE ENGINEERING ---
# Computers don't understand text. We must convert words to numbers.

# A. Skill Flags (One-Hot Encoding for specific skills)
df['Has_Python'] = df['Skills_Detected'].apply(lambda x: 1 if 'Python' in str(x) else 0)
df['Has_SQL'] = df['Skills_Detected'].apply(lambda x: 1 if 'SQL' in str(x) else 0)
df['Has_AWS'] = df['Skills_Detected'].apply(lambda x: 1 if 'AWS' in str(x) else 0)
df['Has_Excel'] = df['Skills_Detected'].apply(lambda x: 1 if 'Excel' in str(x) else 0)

# B. Select Features for Training
# X = Inputs, y = Output (Salary)
features = ['Min_Exp', 'Has_Python', 'Has_SQL', 'Has_AWS', 'Has_Excel']
X = df[features]
y = df['Estimated_Salary']

# --- 4. TRAIN MODEL ---
print("Training Random Forest Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Model Training Complete.")
print(f"Average Error: +/- {mae:.2f} LPA")

# --- 5. SAVE MODEL ---
# We save the model to a file so the Website can use it
with open('salary_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as 'salary_model.pkl'")