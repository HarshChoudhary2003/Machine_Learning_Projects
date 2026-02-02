# Movie Recommendation Based on Emotion 🎭

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-Scraping-green.svg)](https://www.crummy.com/software/BeautifulSoup/)
[![Requests](https://img.shields.io/badge/Requests-HTTP-blueviolet.svg)](https://requests.readthedocs.io/)

Discover movies that match your current mood! This project scrapers IMDb to recommend top-rated feature films based on the user's emotion-to-genre mapping.

## 📌 Project Overview

Stuck on what to watch? Tell the program how you feel, and it will fetch the latest top-rated movies of the corresponding genre directly from IMDb. Whether you're feeling for some high-stakes Action, a deep Drama, or a bone-chilling Horror, this tool has you covered.

## 🛠️ Technologies Used

- **Language:** `Python`
- **HTTP Library:** `Requests`
- **Web Scraping:** `BeautifulSoup (bs4)`
- **Parsing:** `re` (Regular Expressions), `lxml`

## 🧠 How it Works

1.  **Input:** The user enters an emotion (e.g., "Action", "Horror").
2.  **Mapping:** The input is mapped to a specific IMDb search query for that genre.
3.  **Scraping:** The program sends a GET request to IMDb with a modern browser User-Agent.
4.  **Extraction:** `BeautifulSoup` parses the HTML response and extracts movie titles using specific regular expression patterns matching IMDb title links.
5.  **Output:** A curated list of top movie recommendations is displayed.

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Movie recommendation based on emotion"
   ```

2. **Install dependencies:**
   ```bash
   pip install requests beautifulsoup4 lxml
   ```

3. **Run the Script:**
   ```bash
   python main.ipynb
   ```
   *(Or run it directly within a Jupyter Notebook environment)*

## 🤝 Contributing

Contributions are welcome! If you want to add more emotion mappings or improve the scraping logic, feel free to submit a pull request.

---
Developed with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)
