import json
import os

file_path = r"d:\Machine-Learning-Projects\E-Commerce products category classification\main.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["id"] == "b33370db":
        cell["source"] = [
            "import pandas as pd\n",
            "\n",
            "import re\n",
            "import string\n",
            "import nltk\n",
            "from nltk.corpus import stopwords\n",
            "\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.metrics import accuracy_score, classification_report\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        ]
    elif cell["id"] == "39084efa":
        cell["source"] = [
            "def clean_text(text):\n",
            "    text = text.lower()\n",
            "    \n",
            "    # Fix & Enhance: Use translate for faster and accurate punctuation removal\n",
            "    text = text.translate(str.maketrans('', '', string.punctuation))\n",
            "    text = re.sub(r'\\d+', '', text)\n",
            "\n",
            "    # Fix: join words with spaces rather than empty string\n",
            "    text = ' '.join([word for word in text.split() if word not in stop_words])\n",
            "\n",
            "    return text\n",
            "\n",
            "df['cleaned_title'] = df['product_title'].apply(clean_text)\n"
        ]
    elif cell["id"] == "87204886":
        cell["source"] = [
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "\n",
            "# Fix: max_features assignment (was == instead of =)\n",
            "vectorizer = TfidfVectorizer(max_features=5000)\n",
            "# Enhancement: Removed .toarray() to save memory and utilize sparse matrices\n",
            "x = vectorizer.fit_transform(df['cleaned_title'])\n",
            "\n",
            "y = df['category']\n",
            "\n",
            "# Fix: Correct unpacking order for train_test_split\n",
            "xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)\n",
            "\n",
            "# Enhancement: Added max_iter=1000 to prevent ConvergenceWarnings\n",
            "model = LogisticRegression(max_iter=1000)\n",
            "model.fit(xtrain, ytrain)\n"
        ]
    elif cell["id"] == "37ede804":
        cell["source"] = [
            "import pickle\n",
            "\n",
            "# Enhancement: Save both the model and the vectorizer so new inputs can be processed\n",
            "with open('product_model.pkl', 'wb') as model_file:\n",
            "    pickle.dump(model, model_file)\n",
            "\n",
            "with open('product_vectorizer.pkl', 'wb') as vec_file:\n",
            "    pickle.dump(vectorizer, vec_file)\n"
        ]
    elif cell["id"] == "57a9d3dd":
        cell["source"] = [
            "predict = model.predict(xtest)\n",
            "\n",
            "# Fix: Change ytst to ytest\n",
            "print(\"Accuracy Score:\", accuracy_score(ytest, predict))\n",
            "\n",
            "# Enhancement: Add a classification report for detailed metrics\n",
            "print(\"\\nClassification Report:\")\n",
            "print(classification_report(ytest, predict))\n"
        ]
        
    # Clear outputs for cells that had errors
    if cell["id"] in ["87204886", "37ede804", "57a9d3dd"]:
        cell["outputs"] = []

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook fixed successfully!")
