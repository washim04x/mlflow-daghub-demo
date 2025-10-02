from mlflow.tracking import MlflowClient
import mlflow
client=MlflowClient()
mlflow.set_tracking_uri("http://127.0.0.1:5000")   # local tracking server
mlflow.set_experiment('student-dt')
run_id="e0a8a4486a7c4a18b818143b02debbe8"
model_Path="file:C:/Users/ABDUL AZIZ/Desktop/cookicutter/mlflow_model_reg/mlartifacts/1/models/m-490ca667401f46a494f7441db1d35e19/artifacts/MLmodel"

model_uri=f"runs:/{run_id}/{model_Path}"

model_name="rf"
result=mlflow.register_model(model_uri=model_uri, name=model_name)

import time
time.sleep(10)  # wait for a while to let the model be registered
client.update_model_version(
    name=model_name, 
    version=result.version, 
    description="This is the first version of the model"
)

client.set_model_version_tag(
    name=model_name, 
    version=result.version, 
    key="developer", 
    value="washim"
)

