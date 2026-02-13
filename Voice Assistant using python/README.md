# 🎙️ Nova | AI Powered Voice Assistant

<div align="center">
  <img src="https://img.shields.io/badge/AI-Gemini%20Pro-blue?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/Built%20With-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-yellow?style=for-the-badge&logo=python" />
</div>

<br />

**Nova** is a next-generation voice assistant that combines classic automation (opening apps, checking time) with the power of **Google's Gemini AI**. 

Unlike traditional rule-based assistants, Nova can understand context, remember your conversation history, and answer complex questions just like a human.

---

## ✨ Key Features

- **🧠 Advanced AI Engine**: Powered by **Google Gemini 1.5 Flash** for intelligent, context-aware responses.
- **🗣️ Context Awareness**: Remembers previous questions and answers for a natural conversation flow.
- **🎙️ Voice Activated**: Speak naturally to interact, with text-to-speech responses.
- **🎨 Cyberpunk UI**: A stunning, modern interface with neon aesthetics and smooth animations.
- **🛠️ Automation Tools**:
  - 🌐 **Web**: Open Google, YouTube.
  - 📚 **Knowledge**: Integrated Wikipedia search.
  - ⏰ **Utilities**: Time check, jokes.
- **� Secure**: API Key management through the sidebar (keys are never stored).

---

## 🛠️ Technology Stack

- **Core**: `Python 3.x`
- **Frontend**: `Streamlit`
- **AI Model**: `Google Generative AI (Gemini)`
- **Speech**: `SpeechRecognition`, `pyttsx3`, `pyaudio`
- **Utilities**: `wikipedia`, `pyjokes`

---

## 🚀 Installation & Setup

### 1. Prerequisite: Get a Gemini API Key
You'll need a free API key to enable the AI features.
- Go to [Google AI Studio](https://aistudio.google.com/).
- Click **"Get API key"** and create a new key.

### 2. Clone the Repository
```bash
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
cd "Voice Assistant using python"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📖 How to Use

1.  **Launch the App**: Run the command above. A new tab will open in your browser.
2.  **Enter API Key**: In the sidebar (left panel), paste your **Gemini API Key**.
3.  **Activate Voice**: Click the **🔴 ACTIVATE VOICE PROTOCOL** button.
4.  **Speak**:
    *   *"Who was Albert Einstein?"* (Gemini AI)
    *   *"Tell me a joke about programmers"* (PyJokes)
    *   *"What time is it?"* (System Time)
    *   *"Wikipedia Python Programming"* (Wikipedia Search)
5.  **Chat History**: View your entire conversation with Nova in the main chat window.

---

## 🛤️ Future Roadmap
- [ ] Wake word detection ("Hey Nova").
- [ ] IoT Home Automation integration.
- [ ] Multi-language support.
- [ ] Custom voice cloning.

---

## ⚖️ License
Distributed under the MIT License.
