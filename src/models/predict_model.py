import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mlflow

# Paths
cur_dir = Path(__file__)
parent_dir = cur_dir.parent.parent.parent

# Initialize MLflow
# import dagshub
# dagshub.init(repo_owner='washim04x',repo_name='mlflow-daghub-demo',mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/washim04x/mlflow-daghub-demo.mlflow")
mlflow.set_tracking_uri("http://127.0.0.1:5000")   # local tracking server
mlflow.set_experiment('student-dt')

#  use a same run for evaluation
previous_run_id = open(parent_dir / 'models' / 'run_id.txt').read().strip()
with mlflow.start_run(run_id=previous_run_id) as run:
    # Load trained model
    model_path = parent_dir / 'models/model.joblib'
    model = joblib.load(model_path)

    # Load test data
    test_data = pd.read_csv(parent_dir / 'data/processed/test_processed.csv')
    X_test = test_data.drop(columns=['Placed'])
    y_test = test_data['Placed']

    # Predictions and metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric('Accuracy', acc)
    mlflow.log_metric('Precision', prec)
    mlflow.log_metric('Recall', rec)
    mlflow.log_metric('F1', f1)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)
    
    # Log test data as input example
    test_df=mlflow.data.from_pandas(test_data)
    mlflow.log_input(test_df, 'test_data')


    # Set tag
    mlflow.set_tag('author', 'washim')
