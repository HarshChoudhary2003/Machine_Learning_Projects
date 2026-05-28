import os
import json
import re
import string
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import onnxruntime as ort

app = FastAPI(
    title="AI Next-Word Prediction API",
    description="FastAPI backend serving next-word suggestions utilizing an optimized ONNX LSTM model."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.onnx")

# Global variables
vocab = {}
index_to_word = {}
onnx_session = None
prefix_cache = {}  # In-memory dictionary for prefix caching
CONTEXT_LENGTH = 5  # Must match the training sequence length

# Load vocab and ONNX model on startup
@app.on_event("startup")
def startup_event():
    global vocab, index_to_word, onnx_session
    
    # Load vocabulary
    if not os.path.exists(VOCAB_PATH):
        raise RuntimeError(f"Vocabulary file not found at {VOCAB_PATH}. Run training first.")
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    index_to_word = {idx: word for word, idx in vocab.items()}
    
    # Load ONNX model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"ONNX model file not found at {MODEL_PATH}. Run export_onnx.py first.")
    
    # Disable CPU multithreading overhead for single-thread requests
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    onnx_session = ort.InferenceSession(MODEL_PATH, sess_options)
    print("FastAPI: Vocabulary and ONNX model loaded successfully.")

class PredictionRequest(BaseModel):
    text: str
    temperature: float = 1.0
    top_k: int = 3
    use_cache: bool = True

class PredictionResponse(BaseModel):
    suggestions: list[str]
    confidences: list[float]
    from_cache: bool

def clean_text(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global vocab, index_to_word, onnx_session, prefix_cache
    
    cleaned = clean_text(request.text)
    
    # Cache lookup
    cache_key = f"{cleaned}_temp{request.temperature}_k{request.top_k}"
    if request.use_cache and cache_key in prefix_cache:
        cached_val = prefix_cache[cache_key]
        return PredictionResponse(
            suggestions=cached_val["suggestions"],
            confidences=cached_val["confidences"],
            from_cache=True
        )
        
    tokens = cleaned.split()
    
    # Convert to token IDs, mapping unknown words to <unk>
    token_ids = [vocab.get(w, vocab.get("<unk>", 1)) for w in tokens]
    
    # Trim context window or left-pad with <pad> (0)
    if len(token_ids) > CONTEXT_LENGTH:
        token_ids = token_ids[-CONTEXT_LENGTH:]
    else:
        padding = [vocab.get("<pad>", 0)] * (CONTEXT_LENGTH - len(token_ids))
        token_ids = padding + token_ids
        
    # Model inference input shape: [batch_size=1, sequence_length=5]
    input_data = np.array([token_ids], dtype=np.int64)
    
    try:
        # Run ONNX inference
        outputs = onnx_session.run(None, {"input": input_data})
        logits = outputs[0][0]  # Shape: [vocab_size]
        
        # Apply temperature scaling
        if request.temperature > 0.0:
            logits = logits / request.temperature
            
        # Softmax probabilities
        probs = softmax(logits)
        
        # Filter special tokens like <pad> and <unk> from suggestions
        special_idxs = {vocab.get("<pad>", 0), vocab.get("<unk>", 1)}
        for idx in special_idxs:
            probs[idx] = -1.0  # Set probability to negative to ignore it
            
        # Get top-k indices and probabilities
        top_k = min(request.top_k, len(vocab) - len(special_idxs))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        suggestions = []
        confidences = []
        
        for idx in top_indices:
            word = index_to_word.get(int(idx), "")
            confidence = float(probs[idx])
            if word and confidence > 0:
                suggestions.append(word)
                confidences.append(round(confidence * 100, 2))
                
        # Handle case where no valid suggestions are found
        if not suggestions:
            suggestions = ["is", "the", "a"]
            confidences = [33.3, 33.3, 33.3]
            
        # Cache writing
        if request.use_cache:
            prefix_cache[cache_key] = {
                "suggestions": suggestions,
                "confidences": confidences
            }
            
        return PredictionResponse(
            suggestions=suggestions,
            confidences=confidences,
            from_cache=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX Inference Error: {str(e)}")

# Mount static folder for serving web pages
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
@app.get("/")
def read_root():
    static_file_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(static_file_path):
        return FileResponse(static_file_path)
    return {"message": "AI Next-Word Prediction API running. Static files index.html not found."}
