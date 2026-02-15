# Movie Mood Recommender 🎬

A modern, emotionally intelligent movie recommendation app that suggests films based on your current mood. Built with **FastAPI**, **React**, **Tailwind CSS**, and **Framer Motion**.

## ✨ Features

- **Mood-Based Recommendations**: Select from 8 distinct emotions (Happy, Sad, Excited, etc.).
- **Real-Time Scraping**: Fetches the latest top-rated movies directly from IMDb.
- **Modern UI**: 
  - Glassmorphism design
  - Smooth animations with Framer Motion
  - Responsive layout
- **Rich Metadata**: Displays movie posters, ratings, and release years.

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, BeautifulSoup4 (Scraping)
- **Frontend**: React (Vite), Tailwind CSS, Framer Motion, Lucide Icons, Axios

## 🚀 Getting Started

### Prerequisites

- Node.js & npm
- Python 3.8+

### 1. Clone the Repository

```bash
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
cd "Movie recommendation based on emotion"
```

### 2. Backend Setup

Open a terminal and run:

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Start the server
uvicorn backend.main:app --reload
```
The backend will run on `http://localhost:8000`.

### 3. Frontend Setup

Open a **new** terminal and run:

```bash
cd frontend

# Install dependencies (if not already installed)
npm install

# Start the development server
npm run dev
```
The frontend will run on `http://localhost:5173`.

## 📸 Screenshots

*(Add screenshots here after running the app)*

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License.
