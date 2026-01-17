import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


MODEL_DIR = "outputs/models"
EVAL_DIR = "outputs/evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# Confusion Matrix with values
def plot_confusion(cm, class_labels, model_name):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(len(class_labels)), class_labels, rotation=45)
    plt.yticks(range(len(class_labels)), class_labels)

    # Add values inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    path = f"{EVAL_DIR}/{model_name}_confusion.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path


# Main Training Function
def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate:
    - KNN
    - Logistic Regression
    - Random Forest

    Returns:
    - report (dict)
    - train_shape
    - test_shape
    """

    report = {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    class_labels = np.unique(y)

    # -------- Models --------
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )
    }

    #  Training Loop 
    for name, model in models.items():

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro"),
            "Recall": recall_score(y_test, y_pred, average="macro"),
            "F1": f1_score(y_test, y_pred, average="macro")
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        safe_name = name.replace(" ", "_").lower()
        cm_path = plot_confusion(cm, class_labels, safe_name)

        # Save model
        model_path = f"{MODEL_DIR}/{safe_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Report
        report[name] = {
            "metrics": metrics,
            "confusion_matrix": cm_path,
            "model_path": model_path
        }

    return report, X_train.shape, X_test.shape
