import uvicorn
import os

if __name__ == "__main__":
    # Ensure current directory matches script location to resolve paths correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Starting AI Next-Word Prediction System...")
    print("Point your browser to http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
