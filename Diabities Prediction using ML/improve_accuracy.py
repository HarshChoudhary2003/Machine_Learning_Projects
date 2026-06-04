import nbformat as nbf
import json

def generate_notebook():
    with open('main.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # We add new cells to try Random Forest, Gradient Boosting and GridSearchCV
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Improving Accuracy with Ensemble Models\n", "Let's try Random Forest and Gradient Boosting to see if we can get better performance than Logistic Regression."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
                "\n",
                "# Random Forest\n",
                "rf_model = RandomForestClassifier(random_state=42)\n",
                "rf_model.fit(X_train_scaled, y_train)\n",
                "rf_pred = rf_model.predict(X_test_scaled)\n",
                "rf_acc = accuracy_score(y_test, rf_pred)\n",
                "print(f'Random Forest Accuracy: {rf_acc:.4f}')\n",
                "\n",
                "# Gradient Boosting\n",
                "gb_model = GradientBoostingClassifier(random_state=42)\n",
                "gb_model.fit(X_train_scaled, y_train)\n",
                "gb_pred = gb_model.predict(X_test_scaled)\n",
                "gb_acc = accuracy_score(y_test, gb_pred)\n",
                "print(f'Gradient Boosting Accuracy: {gb_acc:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### Hyperparameter Tuning (Grid Search)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.model_selection import GridSearchCV\n",
                "\n",
                "# Define parameter grid for Random Forest\n",
                "param_grid = {\n",
                "    'n_estimators': [50, 100, 200],\n",
                "    'max_depth': [None, 10, 20, 30],\n",
                "    'min_samples_split': [2, 5, 10]\n",
                "}\n",
                "\n",
                "grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), \n",
                "                           param_grid=param_grid, \n",
                "                           cv=5, \n",
                "                           n_jobs=-1, \n",
                "                           scoring='accuracy')\n",
                "                           \n",
                "grid_search.fit(X_train_scaled, y_train)\n",
                "\n",
                "print(\"Best Parameters:\", grid_search.best_params_)\n",
                "best_rf_model = grid_search.best_estimator_\n",
                "\n",
                "# Test best model\n",
                "best_rf_pred = best_rf_model.predict(X_test_scaled)\n",
                "best_rf_acc = accuracy_score(y_test, best_rf_pred)\n",
                "print(f'Tuned Random Forest Accuracy: {best_rf_acc:.4f}')"
            ]
        }
    ]
    
    nb['cells'].extend(new_cells)
    
    with open('main.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    generate_notebook()
    print("Added ensemble models to notebook.")
