import pandas as pd
import numpy as np
import os

def generate_synthetic_data(base_path):
    os.makedirs(base_path, exist_ok=True)

    # 1. Continent Data
    continents = pd.DataFrame({
        'continent_id': [1, 2, 3, 4, 5],
        'continent_name': ['Asia', 'Europe', 'North America', 'South America', 'Africa']
    })
    continents.to_csv(os.path.join(base_path, 'continent_data.csv'), index=False)

    # 2. Region Data
    regions = pd.DataFrame({
        'region_id': [1, 2, 3],
        'region_name': ['East Asia', 'Western Europe', 'North America Coast'],
        'continent_id': [1, 2, 3]
    })
    regions.to_csv(os.path.join(base_path, 'region_data.csv'), index=False)

    # 3. Country Data
    countries = pd.DataFrame({
        'country_id': [1, 2, 3],
        'country_name': ['Japan', 'France', 'USA'],
        'region_id': [1, 2, 3]
    })
    countries.to_csv(os.path.join(base_path, 'country_data.csv'), index=False)

    # 4. City Data
    cities = pd.DataFrame({
        'city_id': [1, 2, 3, 4],
        'city_name': ['Tokyo', 'Paris', 'New York', 'Osaka'],
        'country_id': [1, 2, 3, 1]
    })
    cities.to_csv(os.path.join(base_path, 'city_data.csv'), index=False)

    # 5. Visit Mode Data
    visit_modes = pd.DataFrame({
        'visit_mode_id': [1, 2, 3, 4, 5],
        'visit_mode_name': ['Business', 'Family', 'Couples', 'Solo', 'Friends']
    })
    visit_modes.to_csv(os.path.join(base_path, 'visit_mode_data.csv'), index=False)

    # 6. Attraction Type Data
    attraction_types = pd.DataFrame({
        'attraction_type_id': [1, 2, 3, 4, 5],
        'attraction_type_name': ['Museum', 'Park', 'Beach', 'Historic Site', 'Theme Park']
    })
    attraction_types.to_csv(os.path.join(base_path, 'attraction_type_data.csv'), index=False)

    # 7. Attraction Data
    attractions = pd.DataFrame({
        'attraction_id': range(1, 11),
        'attraction_name': ['Tokyo Tower', 'Louvre', 'Central Park', 'Universal Studios', 'Eiffel Tower', 'Mount Fuji', 'DisneySea', 'Times Square', 'Statue of Liberty', 'Kyoto Shrine'],
        'attraction_type_id': [4, 1, 2, 5, 4, 2, 5, 4, 4, 4],
        'city_id': [1, 2, 3, 4, 2, 1, 4, 3, 3, 4]
    })
    attractions.to_csv(os.path.join(base_path, 'attraction_data.csv'), index=False)

    # 8. User Data
    users = pd.DataFrame({
        'user_id': range(1, 101),
        'user_age': np.random.randint(18, 75, 100),
        'user_gender': np.random.choice(['M', 'F', 'O'], 100),
        'country_id': np.random.choice([1, 2, 3], 100)
    })
    users.to_csv(os.path.join(base_path, 'user_data.csv'), index=False)

    # 9. Transaction Data (Trip Data)
    transactions = pd.DataFrame({
        'user_id': np.random.choice(range(1, 101), 500),
        'attraction_id': np.random.choice(range(1, 11), 500),
        'visit_year': np.random.choice([2021, 2022, 2023, 2024], 500),
        'visit_month': np.random.choice(range(1, 13), 500),
        'rating': np.random.choice([1, 2, 3, 4, 5], 500, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
        'visit_mode_id': np.random.choice(range(1, 6), 500)
    })
    # Add some noise/missing values for cleaning demonstration
    transactions.loc[np.random.choice(transactions.index, 20), 'rating'] = np.nan
    transactions.loc[np.random.choice(transactions.index, 10), 'rating'] = 99 # Outlier
    
    transactions.to_csv(os.path.join(base_path, 'transaction_data.csv'), index=False)

    print(f"Synthetic data generated in {base_path}")

if __name__ == "__main__":
    generate_synthetic_data(r'd:\Machine-Learning-Projects\Tourism_Project\data')
