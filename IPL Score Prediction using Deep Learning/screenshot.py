from playwright.sync_api import sync_playwright
import time
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width":1280, "height":800})
        print("Navigating to app...")
        page.goto("http://localhost:8501")
        
        # Wait for Streamlit to load heavy content
        print("Waiting for .stApp selector...")
        try:
            page.wait_for_selector(".stApp", state="visible", timeout=10000)
            # Give UI animations time to settle (lottie, etc.)
            time.sleep(3)
        except Exception as e:
            print(f"Warning: Selector wait timed out. proceeding anyway. {e}")
            
        print("Taking screenshot...")
        page.screenshot(path="ui.png", full_page=False)
        browser.close()
        print("Screenshot saved to ui.png")

if __name__ == "__main__":
    run()
