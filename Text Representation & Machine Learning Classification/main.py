import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Use non-blocking backend
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# Import modular components from src
try:
    from src.preprocessing import load_dataset, process_dataset, split_data, save_data
    from src.vectorization import tfidf_vectorizer, transform_data
    from src.models import get_models, get_ensemble_model
    from src.evaluation import evaluate_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    print("\n" + "="*50)
    print("🚀 IMDB Sentiment Analysis Advanced Pipeline")
    print("="*50 + "\n")

    # 1. Loading and Preprocessing
    processed_dir = "data/processed"
    processed_files = [
        f"{processed_dir}/X_train.pkl", f"{processed_dir}/y_train.pkl",
        f"{processed_dir}/X_test.pkl", f"{processed_dir}/y_test.pkl"
    ]
    
    if all(os.path.exists(f) for f in processed_files):
        print("📦 Loading preprocessed data...")
        X_train = pickle.load(open(processed_files[0], "rb"))
        y_train = pickle.load(open(processed_files[1], "rb"))
        X_test = pickle.load(open(processed_files[2], "rb"))
        y_test = pickle.load(open(processed_files[3], "rb"))
    else:
        print("🔍 Preprocessed files not found. Starting preprocessing...")
        df = load_dataset()
        df = process_dataset(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        os.makedirs(processed_dir, exist_ok=True)
        save_data(X_train, X_val, X_test, y_train, y_val, y_test)

    # 2. Vectorization
    print(f"\n✨ Vectorizing data (TF-IDF)...")
    vectorizer = tfidf_vectorizer()
    X_train_vec, X_test_vec = transform_data(vectorizer, X_train, X_test)
    print(f"✅ Vectorization complete. Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 3. Models Training and Evaluation
    print("\n🛠️  Initializing individual models...")
    models = get_models()
    results = []
    trained_models = {}
    
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        print(f"\n🏃 Training {name}...")
        start_time = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - start_time
        
        predictions = model.predict(X_test_vec)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, predictions)
        
        results.append({"model": name, "accuracy": acc, "train_time": train_time})
        trained_models[name] = model
        print(f"🏁 {name} Accuracy: {acc:.4f} (Time: {train_time:.2f}s)")

    # 4. Ensemble Exploration
    print("\n🤝 Creating Ensemble Model (Top 3)...")
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    top_3_names = [res['model'] for res in sorted_results[:3]]
    best_models_list = [(name, trained_models[name]) for name in top_3_names]
    
    ensemble = get_ensemble_model(best_models_list)
    start_time = time.time()
    ensemble.fit(X_train_vec, y_train)
    ens_time = time.time() - start_time
    
    ens_preds = ensemble.predict(X_test_vec)
    ens_acc = accuracy_score(y_test, ens_preds)
    
    print(f"🏆 Ensemble Accuracy: {ens_acc:.4f} (Time: {ens_time:.2f}s)")
    results.append({"model": "Ensemble (Top 3)", "accuracy": ens_acc, "train_time": ens_time})
    trained_models["Ensemble"] = ensemble

    # 5. Final Comparison
    results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
    print("\n" + "="*40)
    print("📊 Final Leaderboard")
    print("="*40)
    print(results_df.to_string(index=False))
    
    results_df.to_csv("outputs/reports/model_comparison.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x="accuracy", y="model", data=results_df, palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.xlim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig("outputs/plots/accuracy_comparison.png")
    plt.close()

    # 6. Save the Champion
    best_row = results_df.iloc[0]
    champion_name = best_row["model"]
    champion_model = trained_models.get(champion_name.split(" ")[0], trained_models.get("Ensemble")) if "Ensemble" in champion_name else trained_models[champion_name]
    
    print(f"\n🌟 Saving Champion Model: {champion_name}")
    pickle.dump(champion_model, open("models/best_model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    
    print("\n✅ Pipeline completed successfully! Outputs available in 'outputs/' and 'models/'.")


if __name__ == "__main__":
    main()