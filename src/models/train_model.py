import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
import os
from pathlib import Path
cur_dir=Path(__file__)
parent_dir=cur_dir.parent.parent.parent
params_path=parent_dir.as_posix()+'/params.yaml'
params=yaml.safe_load(open(params_path))['train_model']

train=pd.read_csv(parent_dir.as_posix()+'/data/processed/train_processed.csv')



x_train=train.drop(columns=['Placed'])
y_train=train['Placed']

rf=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],bootstrap=params['bootstrap'],criterion=params['criterion'])
rf.fit(x_train,y_train)

joblib.dump(rf,parent_dir.as_posix()+'/models/model.joblib')
