<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=2,22,25&height=250&section=header&text=Agentic%20AI%20Travel&fontSize=40&animation=fadeIn&fontAlignY=38&desc=Autonomous%20Planning%20with%20LangChain&descAlignY=55&descAlign=50" width="100%" />

  <br/>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![LangChain](https://img.shields.io/badge/LangChain-🦜-green?style=for-the-badge)](https://langchain.com)
  [![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)

---

## 🧳 Overview
This project showcases **Agentic AI** - autonomous systems capable of planning, reasoning, and executing complex tasks. The Travel Planning Assistant uses **LangChain** to orchestrate specialized AI agents that collaborate to deliver high-fidelity, personalized travel itineraries.

### 🌟 Advanced Capabilities
- 🤖 **Autonomous Orchestration** - Multi-agent collaboration logic.
- 🔧 **Dynamic Tooling** - Integration with real-time search and research engines.
- 🌐 **Live Data Ingress** - Gathering current destination insights.
- 📄 **Intelligence Export** - One-click PDF itinerary generation.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Destination Research** | Gather information about places |
| 🏨 **Accommodation Finder** | Suggest hotels and stays |
| ✈️ **Flight Options** | Search for flights |
| 🗓️ **Itinerary Planning** | Day-by-day schedules |
| 💰 **Budget Estimation** | Cost calculations |
| 📄 **PDF Generation** | Export travel plans |

---

## 🏗️ Architecture

### Multi-Agent System
```
┌─────────────────────────────────────────────────────────────┐
│                    TRAVEL PLANNING AGENT                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Research   │  │  Planning   │  │   Budget    │         │
│  │    Agent    │  │    Agent    │  │    Agent    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│                   ┌──────▼──────┐                           │
│                   │  Orchestrator│                          │
│                   │   (LangChain)│                          │
│                   └──────┬──────┘                           │
│                          │                                  │
│  ┌───────────────────────┼───────────────────────┐         │
│  │                       │                       │         │
│  ▼                       ▼                       ▼         │
│ [DuckDuckGo]        [PDF Gen]              [OpenAI/Groq]   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Agentic AI-Based Travel Planning Assistant/
├── app.py               # Streamlit web interface
├── agent.py             # Main AI agent logic
├── pdf_generator.py     # PDF export functionality
├── tools/               # Custom tools for agents
│   ├── search.py
│   ├── calculator.py
│   └── research.py
├── data/                # Reference data
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🚀 Mission Deployment
1. **Clone Repository**:
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Agentic AI-Based Travel Planning Assistant using LangChain"
   ```
2. **Environment Ignition**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **API Configuration**:
   Create a `.env` file with your `OPENAI_API_KEY`.
4. **Launch**:
   ```bash
   streamlit run app.py
   ```

---

## 🔧 How It Works

### 1️⃣ User Input
User provides destination, dates, preferences, and budget.

### 2️⃣ Agent Planning
AI analyzes requirements and creates a task plan.

### 3️⃣ Tool Execution
Agents use tools to gather real-time information.

### 4️⃣ Itinerary Generation
Comprehensive travel plan is created.

### 5️⃣ PDF Export
User can download the complete itinerary.

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| **LangChain** | Agent orchestration |
| **OpenAI/Groq** | LLM backbone |
| **DuckDuckGo** | Web search |
| **Streamlit** | User interface |
| **ReportLab** | PDF generation |

---

## 🌐 API Integration

The system integrates with:
- 🔍 Search engines for research
- 🤖 LLM APIs (OpenAI, Groq)
- 📊 Data APIs for real-time info

---

<div align="center">
  <h3>⭐ If you found this agentic system useful, please consider giving it a star!</h3>
  <p>Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a></p>
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=2,22,25&height=100&section=footer" width="100%" />
</div>
