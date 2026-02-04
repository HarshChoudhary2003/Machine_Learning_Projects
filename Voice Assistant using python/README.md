# 🎙️ Voice Assistant using Python

A modular, lightweight Voice Assistant built with Python that can perform various tasks such as searching Wikipedia, telling jokes, opening websites, and more. This project serves as a great foundation for building your own personalized AI assistant.

---

## ✨ Features

- **🕒 Smart Greetings**: Greets you based on the time of day (Morning, Afternoon, Evening).
- **📚 Wikipedia Integration**: Instantly fetches summaries for any topic from Wikipedia.
- **🤣 Joke Generator**: Lighten the mood with random jokes using `pyjokes`.
- **🌐 Web Automation**: Quick access to Google and YouTube through voice/text commands.
- **⏰ Real-time Clock**: Tells you the current time.
- **🔊 Text-to-Speech**: Crystal clear voice responses using `pyttsx3`.

---

## 🛠️ Technology Stack

- **Python 3.x**
- **Libraries**:
  - `pyttsx3`: For text-to-speech conversion.
  - `SpeechRecognition`: For processing voice commands (extensible).
  - `wikipedia`: For library/data fetching.
  - `pyjokes`: For humor.
  - `webbrowser` & `os`: For system and web interactions.

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Voice Assistant using python"
   ```

2. **Install Dependencies**:
   ```bash
   pip install pyttsx3 SpeechRecognition wikipedia pyjokes pyaudio
   ```

3. **Run the Assistant**:
   Open the `main.ipynb` notebook and execute all cells.

---

## 📖 Usage

Once running, you can typoe or speak commands like:
- *"Wikipedia [Topic]"*
- *"Tell me a joke"*
- *"Open YouTube"*
- *"What is the time?"*
- *"Bye"* or *"Exit"*

---

## 🛤️ Roadmap
- [ ] Implement full Microphone support for voice commands.
- [ ] Integration with GPT/LLM for smarter conversations.
- [ ] Desktop control (Volume, Shutdown, etc.).
- [ ] Email automation.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
