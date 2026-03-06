import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def evaluate_model(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)

    print("Accuracy:", acc)

    print("\nClassification Report")

    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.show()


def plot_roc(y_true, y_prob):

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    auc = roc_auc_score(y_true, y_prob)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")

    plt.plot([0,1],[0,1])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.show()