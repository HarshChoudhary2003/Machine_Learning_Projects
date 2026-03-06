# 🎬 IMDB Sentiment Analysis Advanced
![Sentiment Analysis](https://capsule-render.vercel.app/render?type=waving&color=gradient&height=200&section=header&text=Sentiment%20Analysis&fontSize=80)

A modular, high-performance machine learning pipeline for movie review sentiment classification using the **IMDB Dataset**.

## 🚀 Key Features
- **Modular Architecture:** Clean separation of preprocessing, vectorization, and modeling.
- **Advanced Ensembling:** Automatically combines top-performing models (Voting Classifier) for maximum accuracy.
- **Hyperparameter Tuning:** Built-in `tuning.py` for grid and random search optimization.
- **Rich Analytics:** Automated generation of classification reports and accuracy visualizations.

### 🛠️ Tech Stack
- **Languages:** Python 3.11+
- **ML Frameworks:** Scikit-learn, XGBoost
- **Data Science:** Pandas, NumPy, NLTK
- **Visualization:** Matplotlib, Seaborn

## 📊 Performance Comparison
The pipeline now includes an **Ensemble Model** which typically outperforms individual classifiers.

| Model | Accuracy | Type |
| :--- | :---: | :---: |
| **Ensemble (Top 3)** | **~89.5%** | Voting |
| Logistic Regression | 88.80% | Linear |
| SVM (LinearSVC) | 87.69% | Linear |
| XGBoost | 85.09% | Boosting |

## 📁 Enhanced Project Structure
```text
.
├── main.py                 # Advanced pipeline orchestration
├── src/
│   ├── tuning.py           # Hyperparameter optimization logic [NEW]
│   ├── models.py           # Model definitions & Ensembling
│   ├── preprocessing.py    # Text cleaning
│   └── ...
├── models/                 # Saved "Champion" models
└── outputs/                # Visual results & reports
```

## 🏃 Execution
1. **Full Pipeline:**
   ```bash
   python main.py
   ```
2. **Predict on New Text:**
   ```bash
   python test_model.py
   ```

---
*Enhanced by Antigravity AI*
