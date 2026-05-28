import re
import string
import PyPDF2
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf_stream(pdf_stream):
    """Extracts text from a PDF file stream."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF stream: {e}")
    return text

def preprocess_text(text):
    """Cleans and preprocesses the text."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def extract_entities(text):
    """Extracts entities like email, phone numbers, and names."""
    entities = {}
    
    # 1. Email Extraction using Regex
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = re.findall(email_pattern, text)
    entities['Emails'] = list(set(emails))
    
    # 2. Phone Number Extraction using Regex
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    entities['Phones'] = list(set(phones))
    
    # 3. Name Extraction using spaCy NER
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    entities['Probable Name'] = names[0] if names else "Not Found"
    
    return entities

def calculate_similarity(resume, jd):
    """Calculates cosine similarity between resume and job description."""
    if not resume.strip() or not jd.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume, jd])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def match_skills(resume_text, skills_list):
    """Finds which skills are present in the resume."""
    found_skills = []
    missing_skills = []
    resume_lower = resume_text.lower()
    
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', resume_lower):
            found_skills.append(skill)
        else:
            missing_skills.append(skill)
            
    return found_skills, missing_skills

def analyze_resume(pdf_bytes, custom_jd=None, custom_skills=None):
    """Full pipeline for analyzing a resume."""
    pdf_stream = BytesIO(pdf_bytes)
    resume_text = extract_text_from_pdf_stream(pdf_stream)
    cleaned_resume = preprocess_text(resume_text)
    
    entities = extract_entities(resume_text)
    
    # Default Job Description if none provided
    if not custom_jd:
        custom_jd = """
        We are looking for a Data Scientist or Machine Learning Engineer with strong experience in Python.
        You should be familiar with machine learning frameworks like TensorFlow, Keras, or PyTorch.
        Experience with data analysis, data visualization, Pandas, NumPy, and SQL is required.
        Knowledge of Natural Language Processing (NLP) and Deep Learning is a strong plus.
        """
        
    cleaned_jd = preprocess_text(custom_jd)
    similarity_score = calculate_similarity(cleaned_resume, cleaned_jd)
    
    if not custom_skills:
        custom_skills = ["python", "data analysis", "machine learning", "deep learning", "ai", "nlp", "keras", "data visualization", "data science", "sql", "r", "pandas", "numpy", "pytorch", "tensorflow"]
        
    found_skills, missing_skills = match_skills(resume_text, custom_skills)
    
    return {
        "similarity_score": round(similarity_score * 100, 2),
        "entities": entities,
        "skills_found": found_skills,
        "skills_missing": missing_skills
    }
