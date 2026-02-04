# 🎙️ Speech Recognition using Google Speech API

A Python-based speech-to-text application that leverages the Google Speech Recognition API. This project can transcribe both pre-recorded audio files and live speech from a microphone.

---

## ✨ Features

- **📂 File Transcription**: Supports transcribing `.wav` audio files (and others with additional setup).
- **🎤 Live Recognition**: Real-time speech-to-text using your computer's microphone.
- **🤖 Google API Integration**: Uses Google's powerful cloud-based speech recognition for high accuracy.
- **🔊 Noise Adaptation**: Automatically adjusts for ambient noise to improve recognition quality.

---

## 🛠️ Technology Stack

- **Python 3.x**
- **Libraries**:
  - `SpeechRecognition`: The core library for audio processing.
  - `PyAudio`: Required for microphone input.
  - `pydub`: (Optional) For handling varied audio formats.

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Speech Recognition in Python using Google Speech API"
   ```

2. **Install Dependencies**:
   ```bash
   pip install SpeechRecognition pyaudio pydub
   ```

3. **Run the Project**:
   Open `main.ipynb` and follow the instructions inside.

---

## 📖 Usage

### Option 1: Transcribe a File
Place a `.wav` file in the project folder. The notebook will automatically detect and transcribe it.

### Option 2: Live Microphone
If no file is found, the script will automatically switch to microphone mode.
- Wait for the "Speak now" prompt.
- Talk into your microphone.
- Wait for the transcription to appear!

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
