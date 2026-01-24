# Netflix Movies & TV Shows Analysis & Machine Learning Project 

![Netflix Banner](https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Feature Engineering](#feature-engineering)
7. [Machine Learning Models](#machine-learning-models)
8. [Model Evaluation](#model-evaluation)
9. [Business Impact](#business-impact)
10. [Future Work](#future-work)
11. [Conclusion](#conclusion)


---

## Project Overview
Netflix offers a wide variety of movies and TV shows. The primary goal of this project is to analyze Netflix content, uncover trends, and build machine learning models to predict content categories or types based on features like title, genre, cast, director, and description.  

The project workflow includes:  
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering and text vectorization  
- Model building with ML algorithms  
- Hyperparameter tuning and evaluation  
- Saving the best model for deployment  

This project can help streaming platforms improve content recommendations, optimize genre predictions, and understand viewer trends.

---

## Problem Statement
Netflix hosts thousands of titles, each with various attributes like genre, director, cast, and ratings. The problem is to:  
- Analyze Netflix content trends over time.  
- Predict the type of content (Movie/TV Show) or other categorical features using ML.  
- Optimize content recommendation and content production strategies.

---

## Dataset Description
- **Source:** [Netflix Dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)  
- **Columns:** `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`  
- **Size:** ~5,000 entries  
- **Format:** CSV  

The dataset contains textual, categorical, and numeric features requiring preprocessing.

---

## Data Cleaning & Preprocessing
Steps performed:  
1. Removed duplicates: `df.drop_duplicates()`  
2. Filled missing values for categorical columns:  
```python
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
