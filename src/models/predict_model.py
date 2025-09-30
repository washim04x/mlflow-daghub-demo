import joblib
import pandas as pd
from dvclive import Live
import os
import yaml
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import dagshub
dagshub.init(repo_owner='washim04x',repo_name='mlflow-daghub-demo',mlflow=True)
# import seaborn as sns

cur_dir=Path(__file__)
parent_dir=cur_dir.parent.parent.parent
params_path=parent_dir.as_posix()+'/params.yaml'
params=yaml.safe_load(open(params_path))

mlflow.set_tracking_uri("https://dagshub.com/washim04x/mlflow-daghub-demo.mlflow")
mlflow.set_experiment('student-dt')

with mlflow.start_run():
    
    # Load the trained model
    model = joblib.load(parent_dir.as_posix()+"/models/model.joblib")

    # Load test data
    test_data = pd.read_csv(parent_dir.as_posix()+'/data/processed/test_processed.csv')

    # Assuming the last column is the target
    X_test = test_data.drop(columns=["Placed"])
    y_test = test_data["Placed"]

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metric('Accuracy Score',acc)
    mlflow.log_metric('Precision Score',prec)
    mlflow.log_metric('Recall Score',rec)
    mlflow.log_metric('F1 Score',f1)

    for param,value in params.items():
        for key,value in value.items():
            mlflow.log_param(f'{param}_{key}',value)



    with Live('dvclive',dvcyaml=False) as live:
        live.log_metric('Accuracy Score',acc)
        live.log_metric('Precision Score',prec)
        live.log_metric('Recall Score',rec)
        live.log_metric('F1 Score',f1)

        for param,value in params.items():
            for key,value in value.items():
                live.log_param(f'{param}_{key}',value)


    metric={
        'Accuracy Score' : acc,
        'Precision Score' : prec,
        'Recall Score' : rec,
        'F1 Score' : f1
    }

    with open('metric.json','w') as f:
        json.dump(metric,f,indent=4)




