<div align="center">

# 🧳 Agentic AI Travel Planning Assistant

### *Autonomous Travel Planning with LangChain*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green?style=for-the-badge)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/AI%20Type-Agentic-purple?style=flat-square" />
<img src="https://img.shields.io/badge/Framework-LangChain-blue?style=flat-square" />

---

*An intelligent AI agent that autonomously plans personalized travel itineraries using multiple tools and real-time data.*

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

## 🎯 Overview

This project showcases **Agentic AI** - autonomous AI systems that can plan, reason, and execute complex tasks. The Travel Planning Assistant uses **LangChain** to orchestrate multiple AI agents that work together to create personalized travel plans.

### 🌟 Key Highlights
- 🤖 **Autonomous Planning** - AI agents work independently
- 🔧 **Tool Integration** - Search, calculate, research
- 🌐 **Real-time Data** - Current information
- 📋 **PDF Export** - Download your itinerary

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

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Agentic AI-Based Travel Planning Assistant using LangChain"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
# OPENAI_API_KEY=your_key_here
```

---

## 💻 Usage

### Launch Web App
```bash
streamlit run app.py
```

### Run Agent Directly
```bash
python agent.py
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

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
