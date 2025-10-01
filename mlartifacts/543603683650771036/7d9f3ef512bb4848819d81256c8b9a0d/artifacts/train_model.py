import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import mlflow
import mlflow.data.pandas_dataset

# Paths
cur_dir = Path(__file__)
parent_dir = cur_dir.parent.parent.parent
params_path = parent_dir / 'params.yaml'
params = yaml.safe_load(open(params_path))['train_model']

# Load data
train = pd.read_csv(parent_dir / 'data/processed/train_processed.csv')
x_train = train.drop(columns=['Placed'])
y_train = train['Placed']

# Define model and grid
rf = RandomForestClassifier(random_state=42)
params_grid = {
    'n_estimators': params['n_estimators'],
    'max_depth': params['max_depth'],
    'bootstrap': [params['bootstrap']] if isinstance(params['bootstrap'], bool) else params['bootstrap'],
    'criterion': params['criterion']
}

grid_search = GridSearchCV(estimator=rf, param_grid=params_grid, cv=3, n_jobs=-1, verbose=2)

# MLflow initialization
# import dagshub
# dagshub.init(repo_owner='washim04x',repo_name='mlflow-daghub-demo',mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/washim04x/mlflow-daghub-demo.mlflow")
mlflow.set_tracking_uri("http://127.0.0.1:5000")   # local tracking server
mlflow.set_experiment('student-dt')
# mlflow.sklearn.autolog(log_models=False)  # Disable autologging of models to customize

with mlflow.start_run(description="Best hyperparameter trained rf model") as parent:
    # Train model
    grid_search.fit(x_train, y_train)

    # Log all candidate runs as nested runs
    for i, v in enumerate(grid_search.cv_results_['params']):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(v)
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])

    # Save the best model locally
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, parent_dir / 'models/model.joblib')

    # log the model properly with signature (only once)
    signature = mlflow.models.infer_signature(x_train, best_model.predict(x_train))
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="rf_model",
        signature=signature,
        # logged_model_name="rf_model"   # creates entry in registry
    )


    #log best parameters correctly
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_accuracy", grid_search.best_score_)
    #data
    train_df=mlflow.data.from_pandas(train)
    mlflow.log_input(train_df, 'train_data')

    # Save source code
    mlflow.log_artifact(__file__)

    # Set custom tag
    mlflow.set_tag('author', 'washim')

    # Save run_id locally for reference
    run_id = parent.info.run_id
    with open(parent_dir / 'models' / 'run_id.txt', 'w') as f:
        f.write(run_id)

# Force flush run to tracking server
# mlflow.end_run()
