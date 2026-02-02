# Movie Recommender System 🎬

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-red.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)

This project implements a Movie Recommender System using Correlation-based Collaborative Filtering. It suggests movies similar to a given movie based on user ratings.

## 📌 Project Overview

Recommender systems are common in applications like Netflix, Amazon, and Spotify. This project demonstrates the fundamental concepts of recommendation engines, specifically focusing on building an item-based recommender system that identifies movies showing similar user preference patterns.

## 📈 Methodology

The system works by:
1.  **Data Merging:** Combining movie ratings with movie titles.
2.  **Exploratory Data Analysis (EDA):** Visualizing the distribution of ratings and the number of ratings per movie.
3.  **Matrix Creation:** Creating a pivot table (User-Item Matrix) where each row represents a user and each column a movie.
4.  **Correlation Analysis:** Calculating the correlation between a specific movie's user ratings and all other movies in the matrix using the `corrwith()` method.
5.  **Filtering:** Selecting movies with a significant number of ratings to ensure statistical significance in recommendations.

## 📊 Dataset

- `file.tsv`: Contains user ratings (`user_id`, `item_id`, `rating`, `timestamp`).
- `Movie_Id_Titles.csv`: Maps `item_id` to movie titles.

## 🛠️ Features

- View movie popularity based on number of ratings.
- Calculate average ratings for each movie.
- Recommend top 10 movies similar to classics like *Star Wars (1977)* or *Liar Liar (1997)*.

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Movie Recommender System"
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas matplotlib seaborn
   ```

3. **Explore the Notebook:**
   Launch `main.ipynb` to see the step-by-step implementation and visualization.

## 🤝 Contributing

Contributions are always welcome! Feel free to fork the repo and submit a PR with improvements.

---
Developed with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)
