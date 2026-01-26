import pandas as pd
import re
import sqlite3

def clean_data():
    print("Step 1: Loading Raw Data...")
    try:
        df = pd.read_csv("raw_jobs_final.csv")
        print(f"--> Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("ERROR: raw_jobs_final.csv not found. Did you run the scraper?")
        return

    # --- EXTRACTION FUNCTIONS ---

    def extract_experience(text):
        # Regex logic: Look for "digit - digit Yrs" or "digit Yrs"
        # \d+ means "one or more digits"
        pattern = r"(\d+\s*-\s*\d+\s*Yrs)"
        match = re.search(pattern, str(text), re.IGNORECASE)
        if match:
            return match.group(1)
        return "Not Mentioned"

    def extract_skills(text):
        # We define what we are looking for
        target_skills = ['Python', 'SQL', 'Excel', 'Tableau', 'Power BI', 
                         'Machine Learning', 'Deep Learning', 'AWS', 'Azure', 
                         'Spark', 'Hadoop', 'Java', 'C++', 'TensorFlow', 'PyTorch']
        
        found_skills = []
        text_lower = str(text).lower()
        
        for skill in target_skills:
            # Check if the skill exists in the text (case insensitive)
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return ", ".join(found_skills) if found_skills else "None"

    def extract_location(text):
        # Simple heuristic: Check for major cities
        cities = ['Bangalore', 'Bengaluru', 'Hyderabad', 'Pune', 'Mumbai', 
                  'Delhi', 'Gurgaon', 'Noida', 'Chennai', 'Kolkata']
        
        text_lower = str(text).lower()
        for city in cities:
            if city.lower() in text_lower:
                return city
        return "Remote/Other"

    print("Step 2: Applying Extraction Logic (This is the magic)...")
    
    # Apply the functions to the 'Raw_Blob' column
    # We use .apply(lambda x: ...) to run the function on every single row
    df['Experience_Str'] = df['Raw_Blob'].apply(extract_experience)
    df['Skills_Detected'] = df['Raw_Blob'].apply(extract_skills)
    df['Location_Clean'] = df['Raw_Blob'].apply(extract_location)

    # --- CLEANING EXPERIENCE NUMBERS ---
    # Convert "5 - 8 Yrs" into numbers: Min_Exp = 5, Max_Exp = 8
    def parse_years(exp_str):
        try:
            # Find all numbers in the string
            nums = re.findall(r'\d+', exp_str)
            if len(nums) >= 2:
                return int(nums[0]), int(nums[1])
            elif len(nums) == 1:
                return int(nums[0]), int(nums[0])
            else:
                return 0, 0
        except:
            return 0, 0

    # The zip(*) syntax separates the two returned values into two columns
    df['Min_Exp'], df['Max_Exp'] = zip(*df['Experience_Str'].apply(parse_years))

    # Drop the messy columns to keep it clean
    final_df = df[['Title', 'Company', 'Location_Clean', 'Min_Exp', 'Max_Exp', 'Skills_Detected', 'Raw_Blob']]
    
    print("Step 3: Saving to SQL Database...")
    conn = sqlite3.connect('jobs.db')
    final_df.to_sql('jobs_cleaned', conn, if_exists='replace', index=False)
    conn.close()
    
    # Also save CSV for quick checking
    final_df.to_csv("clean_data.csv", index=False)
    
    print("\nVICTORY: Database 'jobs.db' created.")
    print(final_df.head())

if __name__ == "__main__":
    clean_data()