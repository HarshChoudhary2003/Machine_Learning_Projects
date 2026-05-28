# AI Next-Word Prediction System

A high-performance, real-time next-word prediction system built with PyTorch, ONNX, and FastAPI. 

The system utilizes a Long Short-Term Memory (LSTM) neural network trained to predict the next word in a sequence. To ensure fast inference, the model is exported to ONNX format and served via an optimized FastAPI backend.

## Features

*   **Fast Inference:** Uses ONNX Runtime with optimized threading settings for rapid CPU execution.
*   **Configurable Predictions:** Supports `temperature` scaling and `top_k` sampling to control the diversity and predictability of suggestions.
*   **Real-time API:** Built with FastAPI, providing a robust and documented REST endpoint.
*   **In-Memory Caching:** Automatically caches frequent prefix queries to significantly reduce response times for common phrases.
*   **Interactive Web Interface:** Includes a built-in static frontend (`index.html`) for testing and interacting with the model directly in your browser.

## Project Structure

*   `app/main.py`: The FastAPI application, API endpoints, and ONNX inference logic.
*   `app/static/`: Contains the frontend web interface (`index.html`, `app.js`, `style.css`).
*   `training/`: Contains scripts for training the model (`train.py`), the PyTorch model definition (`model.py`), data preprocessing (`preprocess.py`), and the ONNX exporter (`export_onnx.py`).
*   `run.py`: The main entry point to start the web server.
*   `requirements.txt`: Python dependencies.

## Setup & Installation

1.  **Navigate to the project directory:**
    ```bash
    cd "AI Next-Word Prediction System"
    ```

2.  **Install dependencies:**
    Make sure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python run.py
    ```

4.  **Access the interface:**
    Open your web browser and go to: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API Usage

### `POST /predict`

Get next-word suggestions for a given input text.

**Request Body (JSON):**
```json
{
  "text": "what is the",
  "temperature": 1.0,
  "top_k": 3,
  "use_cache": true
}
```

**Response (JSON):**
```json
{
  "suggestions": ["best", "meaning", "name"],
  "confidences": [45.2, 30.1, 15.5],
  "from_cache": false
}
```
