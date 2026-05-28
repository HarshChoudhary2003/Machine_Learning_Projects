import os
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# Import our backend logic
from analyzer import analyze_resume

app = FastAPI(
    title="Resume Analyzer API",
    description="A powerful backend for analyzing resumes and matching them to job descriptions."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...), 
    job_description: Optional[str] = Form(None),
    skills: Optional[str] = Form(None)
):
    try:
        # Read the file bytes
        contents = await file.read()
        
        custom_skills = None
        if skills:
            # Assume skills are comma-separated
            custom_skills = [s.strip().lower() for s in skills.split(',')]
            
        # Run the analysis pipeline
        result = analyze_resume(contents, custom_jd=job_description, custom_skills=custom_skills)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mount static folder for serving web pages
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    
app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
@app.get("/")
def read_root():
    static_file_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(static_file_path):
        return FileResponse(static_file_path)
    return {"message": "Resume Analyzer API running. Static files not found."}

if __name__ == "__main__":
    import uvicorn
    # Change dir to script location
    os.chdir(BASE_DIR)
    print("Starting Resume Analyzer UI on http://127.0.0.1:8001")
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
