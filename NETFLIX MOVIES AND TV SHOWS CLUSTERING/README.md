<div align="center">

# 🎬 Netflix Movies and TV Shows Clustering

### *Unsupervised Learning for Content Recommendation*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/ML%20Type-Clustering-orange?style=flat-square" />
<img src="https://img.shields.io/badge/Platform-Netflix-E50914?style=flat-square" />

---

*Cluster Netflix content to build a recommendation system using unsupervised machine learning.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Results](#-results)
- [Applications](#-applications)

---

## 🎯 Overview

Netflix has thousands of movies and TV shows. This project uses **unsupervised learning** to cluster similar content, enabling personalized recommendations and content discovery.

### 🌟 Objectives
- 🎬 Cluster similar content together
- 📊 Analyze content distribution
- 🔍 Discover hidden patterns
- 🎯 Build recommendation foundation

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **Files** | `NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv` |
| **Size** | ~3 MB |
| **Content** | Netflix catalog metadata |

### Features Analyzed
- 🎬 Title, Type (Movie/TV Show)
- 📅 Release year
- 🌍 Country of origin
- 🎭 Genre/Listed in
- ⏱️ Duration
- 📝 Description

---

## 🔬 Methodology

### Clustering Pipeline
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Load     │ --> │   Text Process  │ --> │   Vectorize     │
│                 │     │   (NLP)         │     │   (TF-IDF)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐     ┌───────▼─────────┐
│   Visualize     │ <-- │   Clustering    │ <-- │   Dim Reduce    │
│   Clusters      │     │   (K-Means)     │     │   (PCA)         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Techniques Used
- 📝 **NLP** - Text preprocessing
- 📊 **TF-IDF** - Feature extraction
- 📉 **PCA** - Dimensionality reduction
- 🎯 **K-Means** - Clustering algorithm
- 🌳 **Hierarchical** - Alternative clustering

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/NETFLIX MOVIES AND TV SHOWS CLUSTERING"

# Install dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter
```

---

## 💻 Usage

```bash
# EDA Notebook
jupyter notebook Sample_EDA_Submission_Template.ipynb

# ML Notebook
jupyter notebook Sample_ML_Submission_Template.ipynb
```

---

## 📈 Results

### Content Distribution
```
┌────────────────────────────────────────────┐
│           NETFLIX CONTENT TYPE             │
├────────────────────────────────────────────┤
│  🎬 Movies:    ████████████████████  70%   │
│  📺 TV Shows:  ████████              30%   │
└────────────────────────────────────────────┘
```

### Cluster Insights
| Cluster | Theme | Example Content |
|---------|-------|-----------------|
| 1 | Action/Adventure | Action movies |
| 2 | Drama/Romance | Romantic films |
| 3 | Documentary | Documentaries |
| 4 | Kids/Family | Children's content |

---

## 💡 Applications

| Use Case | Description |
|----------|-------------|
| 🎯 **Recommendations** | Suggest similar content |
| 📊 **Content Strategy** | Identify gaps |
| 🔍 **Discovery** | Help users find content |
| 📈 **Analytics** | Understand content mix |

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
