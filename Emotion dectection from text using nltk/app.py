from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Initialize the HuggingFace emotion detection pipeline
print("Loading model...")
# Using j-hartmann/emotion-english-distilroberta-base which classifies into 7 emotions:
# anger, disgust, fear, joy, neutral, sadness, surprise
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
print("Model loaded successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Run the pipeline
    prediction = classifier(text)[0]
    
    # The pipeline returns {'label': 'emotion', 'score': 0.95}
    return jsonify({
        'emotion': prediction['label'],
        'score': prediction['score']
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
