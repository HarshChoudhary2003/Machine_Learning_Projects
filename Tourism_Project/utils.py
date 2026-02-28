import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(data_path):
    # Load all tables
    transaction_df = pd.read_csv(os.path.join(data_path, 'transaction_data.csv'))
    user_df = pd.read_csv(os.path.join(data_path, 'user_data.csv'))
    city_df = pd.read_csv(os.path.join(data_path, 'city_data.csv'))
    country_df = pd.read_csv(os.path.join(data_path, 'country_data.csv'))
    region_df = pd.read_csv(os.path.join(data_path, 'region_data.csv'))
    continent_df = pd.read_csv(os.path.join(data_path, 'continent_data.csv'))
    attraction_df = pd.read_csv(os.path.join(data_path, 'attraction_data.csv'))
    att_type_df = pd.read_csv(os.path.join(data_path, 'attraction_type_data.csv'))
    visit_mode_df = pd.read_csv(os.path.join(data_path, 'visit_mode_data.csv'))

    # 1. Handle missing values
    transaction_df['rating'] = transaction_df['rating'].fillna(transaction_df['rating'].median())
    
    # 2. Standardize categorical variables
    city_df['city_name'] = city_df['city_name'].str.strip().str.title()
    visit_mode_df['visit_mode_name'] = visit_mode_df['visit_mode_name'].str.strip().str.title()

    # 3. Remove duplicates
    transaction_df = transaction_df.drop_duplicates()

    # 4. Handle outliers in Rating (1-5)
    transaction_df.loc[transaction_df['rating'] > 5, 'rating'] = 5
    transaction_df.loc[transaction_df['rating'] < 1, 'rating'] = 1

    # 5. Handle datetime
    # We will keep Year and Month as separate features but could also create a visit_date
    transaction_df['visit_date'] = pd.to_datetime({'year': transaction_df['visit_year'], 
                                                  'month': transaction_df['visit_month'], 
                                                  'day': 1})

    # 6. Merge tables
    # First merge attraction and attraction type
    attraction_full = pd.merge(attraction_df, att_type_df, on='attraction_type_id', how='left')
    
    # Merge city, country, region, continent
    location_df = pd.merge(city_df, country_df, on='country_id', how='left')
    location_df = pd.merge(location_df, region_df, on='region_id', how='left')
    location_df = pd.merge(location_df, continent_df, on='continent_id', how='left')
    
    # Add location to attraction
    attraction_full = pd.merge(attraction_full, location_df, on='city_id', how='left')
    
    # Merge user and country
    user_full = pd.merge(user_df, country_df, on='country_id', how='left', suffixes=('_user', '_location'))
    
    # Merge everything into transaction
    df = pd.merge(transaction_df, user_full, on='user_id', how='left')
    df = pd.merge(df, attraction_full, on='attraction_id', how='left', suffixes=('', '_att'))
    df = pd.merge(df, visit_mode_df, on='visit_mode_id', how='left')

    return df

def feature_engineering(df):
    # Aggregated user features
    user_stats = df.groupby('user_id').agg({
        'rating': ['mean', 'count'],
        'visit_mode_name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Solo'
    }).reset_index()
    user_stats.columns = ['user_id', 'avg_user_rating', 'total_user_visits', 'freq_visit_mode']
    df = pd.merge(df, user_stats, on='user_id', how='left')

    # Aggregated attraction features
    att_stats = df.groupby('attraction_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    att_stats.columns = ['attraction_id', 'avg_attraction_rating', 'popularity_score']
    df = pd.merge(df, att_stats, on='attraction_id', how='left')

    # Encoding
    le = LabelEncoder()
    categorical_cols = ['visit_mode_name', 'continent_name', 'country_name', 'attraction_type_name']
    for col in categorical_cols:
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))

    return df

def prepare_models_data(df):
    # Select features
    features = ['user_age', 'avg_user_rating', 'total_user_visits', 'avg_attraction_rating', 
                'popularity_score', 'continent_name_encoded', 'country_name_encoded', 
                'attraction_type_name_encoded', 'visit_year', 'visit_month']
    
    X = df[features]
    y_reg = df['rating']
    y_clf = df['visit_mode_id']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_reg, y_clf, features, scaler
