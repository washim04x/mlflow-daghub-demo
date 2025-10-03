from mlflow.tracking import MlflowClient
import mlflow
import time

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("student-dt")
client = MlflowClient()

run_id = "ed042e7351cc41329407435b13cfc181"
model_uri = f"runs:/{run_id}/rf_model"
model_name = "Rf"

# # 1️⃣ Register model
# result = mlflow.register_model(model_uri=model_uri, name=model_name)

# # 2️⃣ Wait until registration is visible
# time.sleep(20)  # Windows file-store may need some time

# 3️⃣ Transition to Production
client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Archived",
    archive_existing_versions=False  # safely archive any existing Production version
)

print(f"Model {model_name} version {2} is now in Production.")
