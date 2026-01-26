import time
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_mass_jobs(pages_to_scrape=10):
    print("--- STARTING MASS SCRAPE ---")
    
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
    # options.add_argument("--headless") # Keep commented out so you can watch it work
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    all_jobs = []
    
    # Loop through pages 1 to 10
    for page in range(1, pages_to_scrape + 1):
        print(f"\n[Page {page}/{pages_to_scrape}] Loading...")
        
        # URL with pagination
        url = f"https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&txtKeywords=Data+Scientist&sequence={page}&startPage=1"
        
        driver.get(url)
        
        # Random sleep to look human (between 3 and 6 seconds)
        time.sleep(random.uniform(3, 6))
        
        try:
            wait = WebDriverWait(driver, 10)
            
            # Use the ANCHOR method (Find 'View Details' buttons)
            buttons = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//*[contains(text(), 'View Details')]")))
            
            print(f"--> Found {len(buttons)} jobs on this page.")
            
            for btn in buttons:
                try:
                    # Climb up 3 levels to the Card
                    card = btn.find_element(By.XPATH, "./../../..")
                    
                    # Extract text
                    full_text = card.text
                    lines = [line for line in full_text.split('\n') if line.strip() != '']
                    
                    all_jobs.append({
                        "Raw_Blob": full_text, # We keep the mess to clean later
                        "Title": lines[0] if len(lines) > 0 else "N/A",
                        "Company": lines[1] if len(lines) > 1 else "N/A"
                    })
                except:
                    continue
                    
        except Exception as e:
            print(f"Error on page {page} (Might be end of results): {e}")
            break # If error, stop the loop

    driver.quit()
    
    print(f"\n--- SCRAPE FINISHED ---")
    print(f"Total Jobs Collected: {len(all_jobs)}")
    
    # Save to CSV
    df = pd.DataFrame(all_jobs)
    df.to_csv("raw_jobs_final.csv", index=False)
    print("Data saved to 'raw_jobs_final.csv'")

if __name__ == "__main__":
    scrape_mass_jobs()