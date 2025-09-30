import pandas as pd

from sklearn.model_selection import train_test_split
import os
import yaml
from pathlib import Path

url='https://raw.githubusercontent.com/campusx-official/toy-datasets/main/student_performance.csv'
df=pd.read_csv(url)
cur_dir=Path(__file__)
parent_dir=cur_dir.parent.parent.parent
params_path=parent_dir.as_posix()+'/params.yaml'
params=yaml.safe_load(open(params_path))['data_ingetion']

train,test=train_test_split(df,test_size=params['test_size'],random_state=params['random_state'])

raw_path=parent_dir.as_posix()+'/data/raw'

train.to_csv(raw_path+'/train.csv',index=False)
test.to_csv(raw_path+'/test.csv',index=False)

