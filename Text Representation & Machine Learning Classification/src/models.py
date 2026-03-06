from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def get_models():
    models = {
        "MultinomialNB": MultinomialNB(alpha=1.0),
        "BernoulliNB": BernoulliNB(alpha=1.0),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000
        ),
        "SVM": LinearSVC(
            C=1.0,
            max_iter=5000
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    return models

def get_ensemble_model(best_models_list):
    """
    Creates a soft voting classifier from a list of (name, model) tuples.
    Note: LinearSVC doesn't support predict_proba by default, so we might need hard voting or CalibratedClassifierCV.
    """
    return VotingClassifier(estimators=best_models_list, voting='hard')
