<div align="center">

# 💼 SkillSync Pro

### *Intelligent Job Market Analyzer*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Type-Web%20App-blue?style=flat-square" />
<img src="https://img.shields.io/badge/ML-Salary%20Prediction-purple?style=flat-square" />

---

*An AI-powered job market intelligence platform that scrapes, analyzes, and predicts salary trends.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview

**SkillSync Pro** is a comprehensive job market analyzer that combines web scraping, data analysis, and machine learning to provide actionable insights about the job market. It helps job seekers understand salary trends and make informed career decisions.

### 🌟 Key Highlights
- 🕷️ **Real-Time Scraping** - Fresh job data collection
- 🧹 **Data Cleaning** - Automated preprocessing pipeline
- 💰 **Salary Prediction** - ML-powered salary estimation
- 📊 **Interactive Dashboard** - Beautiful Streamlit interface

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Job Scraping** | Automated job listing collection |
| 🧽 **Data Cleaning** | Remove duplicates, normalize data |
| 📈 **Market Analysis** | Trends, distributions, insights |
| 💵 **Salary Model** | Predict salaries based on skills |
| 📊 **Visualization** | Interactive charts and graphs |
| 💾 **Database Storage** | SQLite for data persistence |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SKILLSYNC PRO                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Scraper  │ -> │ Cleaner  │ -> │ Database │              │
│  │ (scraper │    │ (cleaner │    │ (jobs.db)│              │
│  │   .py)   │    │   .py)   │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                        │                    │
│                                        ▼                    │
│  ┌──────────────────────────────────────────┐              │
│  │              ML Model                    │              │
│  │          (model_train.py)                │              │
│  │        salary_model.pkl                  │              │
│  └──────────────────────────────────────────┘              │
│                        │                                    │
│                        ▼                                    │
│  ┌──────────────────────────────────────────┐              │
│  │           Streamlit App                  │              │
│  │              (app.py)                    │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
SkillSync/
├── app.py                  # Streamlit web application
├── scraper.py              # Job scraping module
├── cleaner.py              # Data cleaning pipeline
├── model_train.py          # ML model training
├── salary_model.pkl        # Trained salary prediction model
├── jobs.db                 # SQLite database
├── clean_data.csv          # Processed job data
├── raw_jobs_final.csv      # Raw scraped data
├── day1_anchor_data.csv    # Reference dataset
├── requirements.txt        # Dependencies
└── SkillSync-Pro-*.pdf     # Project documentation
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/SkillSync"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Run the Web App
```bash
streamlit run app.py
```

### Scrape New Data
```bash
python scraper.py
```

### Clean Data
```bash
python cleaner.py
```

### Train Model
```bash
python model_train.py
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML** | Scikit-Learn |
| **Database** | SQLite |
| **Scraping** | BeautifulSoup/Selenium |
| **Data** | Pandas, NumPy |

---

## 📈 Model Performance

The salary prediction model is trained on collected job data and provides estimates based on:
- 📚 Required skills
- 🏢 Company size
- 📍 Location
- 💼 Experience level

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
