
import streamlit as st
import speech_recognition as sr
import pyttsx3
import datetime
import wikipedia
import webbrowser
import os
import pyjokes
import time
import requests
import json
from streamlit_lottie import st_lottie
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nova | AI Powered Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie Animations
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=2)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_voice = load_lottieurl("https://lottie.host/6b3e8954-6e16-4302-8663-888496739768/p5T2q3a1c8.json") # Voice wave
lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_mjpbd2z3.json") # Robot/AI

# Initialize Text-to-Speech Engine
def speak(text):
    try:
        engine = pyttsx3.init()
        # Set properties (optional)
        voices = engine.getProperty('voices')
        # Try to set a female voice if available, else default
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id) 
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"TTS Error: {e}")

# Speech Recognition Function
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now 🎙️")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            st.success("Processing...")
            query = r.recognize_google(audio, language='en-in')
            return query.lower()
        except sr.WaitTimeoutError:
            return "timeout"
        except sr.UnknownValueError:
            return "unknown"
        except Exception as e:
            return f"error: {e}"

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING (Cyberpunk/Dark Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto Mono', monospace;
    }
    
    .stApp {
        background-color: #050505;
        background-image: linear-gradient(0deg, transparent 24%, rgba(255, 255, 255, .05) 25%, rgba(255, 255, 255, .05) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, .05) 75%, rgba(255, 255, 255, .05) 76%, transparent 77%, transparent), linear-gradient(90deg, transparent 24%, rgba(255, 255, 255, .05) 25%, rgba(255, 255, 255, .05) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, .05) 75%, rgba(255, 255, 255, .05) 76%, transparent 77%, transparent);
        background-size: 50px 50px;
    }
    
    /* Neon Text */
    h1, h2, h3 {
        color: #00ffcc !important;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc;
    }
    
    /* Chat Bubble */
    .user-msg {
        background: #1e1e1e;
        color: #fff;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        text-align: right;
        border: 1px solid #333;
        float: right;
        clear: both;
        max-width: 80%;
    }
    .bot-msg {
        background: rgba(0, 255, 204, 0.1);
        color: #00ffcc;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        text-align: left;
        border: 1px solid #00ffcc;
        float: left;
        clear: both;
        max-width: 80%;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.2);
    }
    
    /* Button */
    div.stButton > button {
        background: transparent;
        color: #00ffcc;
        border: 2px solid #00ffcc;
        border-radius: 5px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: 0.3s;
        width: 100%;
        padding: 15px;
    }
    div.stButton > button:hover {
        background: #00ffcc;
        color: #000;
        box-shadow: 0 0 20px #00ffcc;
    }
    
    /* Container/Sidebar */
    .sidebar .sidebar-content {
        background-color: #0a0a0a;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. APP LOGIC
# -----------------------------------------------------------------------------

# Initialize Chat History
if "history" not in st.session_state:
    st.session_state.history = []

def process_command(query, api_key=None):
    response = ""
    
    if 'wikipedia' in query:
        response = "Searching Wikipedia..."
        try:
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            response = f"According to Wikipedia: {results}"
        except:
            response = "Error finding Wikipedia page."
            
    elif 'open youtube' in query:
        response = "Opening YouTube..."
        webbrowser.open("https://www.youtube.com/")
        
    elif 'open google' in query:
        response = "Opening Google..."
        webbrowser.open("https://www.google.com/")
        
    elif 'time' in query:
        strTime = datetime.datetime.now().strftime("%H:%M:%S")
        response = f"The current time is {strTime}"
        
    elif 'joke' in query:
        response = pyjokes.get_joke()
        
    elif 'who are you' in query:
        response = "I am Nova, your advanced AI voice assistant."
        
    else:
        # Fallback to Gemini AI
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Format history for Gemini context
                gemini_history = []
                # Skip the last message which is the current query (we send it separately)
                for msg in st.session_state.history[:-1]:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append({"role": role, "parts": [msg["text"]]})
                
                chat = model.start_chat(history=gemini_history)
                
                # Nova Persona Prompt
                prompt = f"You are Nova, an advanced AI assistant. Answer the following concisely: {query}"
                
                ai_response = chat.send_message(prompt)
                response = ai_response.text
            except Exception as e:
                response = f"AI Error: {str(e)}"
        else:
            response = "I can answer that, but I need a Google Gemini API Key. Please enter it in the sidebar."
        
    return response

# -----------------------------------------------------------------------------
# 4. UI LAYOUT
# -----------------------------------------------------------------------------

# Sidebar for Settings
with st.sidebar:
    st.title("⚙️ SETTINGS")
    st.markdown("Enable Advanced AI capabilities by providing a Gemini API Key.")
    
    # API Key Input
    gemini_key = st.text_input("Gemini API Key", type="password", help="Get your free key at aistudio.google.com")
    
    if gemini_key:
        st.success("API Key Provided! ✅")
    else:
        st.warning("No API Key found. Basic commands only.")
        
    st.markdown("---")
    st.markdown("### 📝 COMMANDS")
    st.markdown("- 'Time' - Current time")
    st.markdown("- 'Wikipedia [Topic]' - Search wiki")
    st.markdown("- 'Open Google/YouTube'")
    st.markdown("- 'Tell me a joke'")
    st.markdown("- **NEW:** Ask anything else! (Requires Key)")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## ⚡ SYSTEM STATUS: ONLINE")
    if lottie_ai:
        st_lottie(lottie_ai, height=250, key="ai_anim")
    
    st.markdown("### 🎙️ COMMAND CENTER")
    st.info("Click 'Activate Voice Protocol' to speak.")
    
    if st.button("🔴 ACTIVATE VOICE PROTOCOL"):
        user_query = listen()
        
        if user_query == "timeout":
            st.warning("⏱️ Listening timed out.")
        elif user_query == "unknown":
            st.error("❌ Could not understand audio.")
        elif "error" in user_query:
            st.error(f"⚠️ {user_query}")
        else:
            # Add User Query to History (Capitalize for display)
            st.session_state.history.append({"role": "user", "text": user_query.capitalize()})
            
            # Process & Speak
            bot_response = process_command(user_query, gemini_key)
            st.session_state.history.append({"role": "bot", "text": bot_response})
            speak(bot_response)

with col2:
    st.title("NOVA /// VOICE AI")
    st.markdown("---")
    
    # Chat Display
    chat_placeholder = st.empty()
    
    with chat_placeholder.container():
        for chat in st.session_state.history:
            if chat["role"] == "user":
                st.markdown(f'<div class="user-msg">🧑‍💻 {chat["text"]}</div><div style="clear:both;"></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">🤖 {chat["text"]}</div><div style="clear:both;"></div>', unsafe_allow_html=True)

    # Empty State
    if not st.session_state.history:
        st.markdown("""
        <div style='text-align: center; color: #555; margin-top: 100px;'>
            <h3>AWAITING INPUT...</h3>
            <p>Try saying: "Tell me a joke", "Time check", or "Wikipedia Python"</p>
        </div>
        """, unsafe_allow_html=True)
