from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import os
import numpy as np

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")), name="static")

# Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
META_PATH = os.path.join(BASE_DIR, "model", "meta_data.pkl")

model = None
meta_data = None

def load_artifacts():
    global model, meta_data
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(META_PATH):
        meta_data = joblib.load(META_PATH)

load_artifacts()

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/meta")
def get_meta():
    if meta_data is None:
        load_artifacts()
        if meta_data is None:
            return {"error": "Model not trained yet"}
    
    # Process metadata for frontend
    features = meta_data['features']
    stats = meta_data['feature_stats']
    encoders = meta_data.get('encoders', {})
    
    # Create schema for frontend
    schema = []
    for feat in features:
        feat_info = {
            "name": feat,
            "label": feat.replace("_", " ").title(),
            "type": "number"
        }
        
        if feat in encoders:
            feat_info["type"] = "select"
            # Get classes from encoder
            classes = encoders[feat].classes_.tolist()
            feat_info["options"] = classes
        else:
            # Add min/max/mean for numbers to help UI
            if feat in stats:
                # Sanitize NaN/Inf values for JSON compliance
                def sanitary_float(val):
                    if isinstance(val, (int, float)):
                        if np.isnan(val) or np.isinf(val):
                            return None
                    return val

                feat_info["min"] = sanitary_float(stats[feat].get('min'))
                feat_info["max"] = sanitary_float(stats[feat].get('max'))
                feat_info["default"] = sanitary_float(stats[feat].get('mean'))
                
        schema.append(feat_info)
        
    return {"schema": schema}

@app.post("/api/predict")
async def predict(request: Request):
    if model is None:
        return {"error": "Model not loaded"}
    
    data = await request.json()
    
    # Prepare input dataframe
    # We need to maintain the same order as training
    feature_order = meta_data['features']
    input_data = {}
    
    for feat in feature_order:
        val = data.get(feat)
        
        # Handle categorical encoding if needed
        if feat in meta_data['encoders']:
            encoder = meta_data['encoders'][feat]
            # Handle potential unseen labels or just raw values if passed correctly
            # Assuming frontend sends the original string value
            try:
                # If value is string and encoder expects string
                if val in encoder.classes_:
                    val = encoder.transform([val])[0]
                else:
                    # Fallback or error
                    val = 0 # Default to 0 index if unknown
            except:
                val = 0
        else:
            # Ensure numerical
            try:
                val = float(val)
            except:
                val = 0.0
        
        input_data[feat] = val
        
    df_input = pd.DataFrame([input_data])
    
    # Predict
    prediction = model.predict(df_input)
    
    return {"prediction": float(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
