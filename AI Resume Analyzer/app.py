from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import fitz
from sentence_transformers import SentenceTransformer, util
from utils import extract_entities
import re

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_pymupdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_file = request.files['resume']
    job_desc = request.form['job_description']

    if not resume_file or not job_desc:
        return "Resume file or job description missing.", 400

    filename = secure_filename(resume_file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    resume_file.save(save_path)

    resume_text = extract_text_pymupdf(save_path)

    embeddings = model.encode([resume_text, job_desc], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    match_score = round(similarity_score * 100, 2)

    resume_entities = extract_entities(resume_text)
    resume_skills = set()
    for label in ['SKILL', 'ORG', 'MISC']:
        resume_skills.update(w.lower() for w in resume_entities.get(label, []))

    jd_words = set(re.findall(r'\b\w{4,}\b', job_desc.lower()))
    matched_skills = sorted(jd_words & resume_skills)
    missing_skills = sorted(jd_words - resume_skills)[:10]

    # Rule-based suggestions
    suggestions = []
    if "flask" in jd_words and "flask" not in resume_text.lower():
        suggestions.append("Mention your experience working with Flask or similar web frameworks.")
    if "sql" in jd_words and "sql" not in resume_text.lower():
        suggestions.append("Include your experience with SQL databases.")
    if "api" in jd_words and "api" not in resume_text.lower():
        suggestions.append("Describe any projects where you built or worked with APIs.")
    if len(resume_text.split()) < 150:
        suggestions.append("Your resume seems short. Add more details about your experience, skills, or projects.")
    if not any(heading in resume_text.lower() for heading in ["experience", "work", "project"]):
        suggestions.append("Consider adding an 'Experience' or 'Projects' section to highlight your background.")

    return render_template(
        'result.html',
        match_score=match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        suggestions=suggestions
    )

if __name__ == '__main__':
    app.run(debug=True)
