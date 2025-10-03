from mlflow.tracking import MlflowClient
import mlflow
client=MlflowClient()
mlflow.set_tracking_uri("http://127.0.0.1:5000")   # local tracking server
mlflow.set_experiment('student-dt')
run_id="ed042e7351cc41329407435b13cfc181"
model_Path="file:file:C:/Users/ABDUL AZIZ/Desktop/cookicutter/mlflow_model_reg/mlartifacts/1/models/m-bdc6a7a5521147678defc1f6d2fc2947/artifacts/MLmodel"

model_uri=f"runs:/{run_id}/rf_model"

model_name="Rf"
result=mlflow.register_model(model_uri=model_uri, name=model_name)

import time
time.sleep(10)  # wait for a while to let the model be registered

# client.update_model_version(
#     name=model_name, 
#     version=result.version, 
#     description="This is the first version of the model"
# )

# client.set_model_version_tag(
#     name=model_name, 
#     version=result.version, 
#     key="developer", 
#     value="washim"
# )

