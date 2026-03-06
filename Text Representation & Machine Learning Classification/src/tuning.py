from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

def tune_logistic_regression(X_train, y_train):
    print("Tuning Logistic Regression...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    }
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Params: {grid.best_params_}")
    return grid.best_estimator_

def tune_xgb(X_train, y_train):
    print("Tuning XGBoost...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    random_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 
                                       param_dist, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    print(f"Best Params: {random_search.best_params_}")
    return random_search.best_estimator_
