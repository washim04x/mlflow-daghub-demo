import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import dagshub
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import mlflow

dagshub.init(repo_owner='washim04x', repo_name='mlflow-daghub-demo', mlflow=True)

# Correct order + sklearn-specific autolog
mlflow.set_tracking_uri("https://dagshub.com/washim04x/mlflow-daghub-demo.mlflow")
mlflow.set_experiment("student-dt")
mlflow.sklearn.autolog(log_models=True)   # <-- important

cur_dir = Path(__file__)
parent_dir = cur_dir.parent.parent.parent
params_path = parent_dir / "params.yaml"
params = yaml.safe_load(open(params_path))["train_model"]

with mlflow.start_run():
    # Load train/test
    train = pd.read_csv(parent_dir / "data/processed/train_processed.csv")
    test = pd.read_csv(parent_dir / "data/processed/test_processed.csv")

    X_train = train.drop(columns=["Placed"])
    y_train = train["Placed"]

    rf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        bootstrap=params["bootstrap"],
        criterion=params["criterion"]
    )
    rf.fit(X_train, y_train)  # autolog hooks here

    joblib.dump(rf, parent_dir / "models/model.joblib")

    # Test
    X_test = test.drop(columns=["Placed"])
    y_test = test["Placed"]
    y_pred = rf.predict(X_test)

    # Manual metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log them manually
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")   # <-- log the plot

    # Tag
    mlflow.set_tag("author", "washim")
