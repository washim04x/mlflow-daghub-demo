import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yaml
import os
from pathlib import Path

cur_dir=Path(__file__)
parent_dir=cur_dir.parent.parent.parent
params_path=parent_dir.as_posix()+'/params.yaml'
params=yaml.safe_load(open(params_path))['feature_eg']

train=pd.read_csv(parent_dir.as_posix()+'/data/raw/train.csv')
test=pd.read_csv(parent_dir.as_posix()+'/data/raw/test.csv')



x_train=train.drop(columns=['Placed'])
y_train=train['Placed']
x_test=test.drop(columns=['Placed'])
y_test=test['Placed']

scaler=StandardScaler()
x_train_sc=scaler.fit_transform(x_train)
x_test_sc=scaler.transform(x_test)

pca=PCA(n_components=params['n_components'])
x_train_pca=pca.fit_transform(x_train_sc)
x_test_pca=pca.fit_transform(x_test_sc)

train_processed=pd.DataFrame(x_train_pca,columns=[f'PC{i+1}' for i in range(params['n_components'])])
train_processed['Placed']=y_train.reset_index(drop=True)
test_processed=pd.DataFrame(x_test_pca,columns=[f'PC{i+1}' for i in range(params['n_components'])])
test_processed['Placed']=y_test.reset_index(drop=True)


train_processed.to_csv(parent_dir.as_posix()+'/data/processed/train_processed.csv',index=False)
test_processed.to_csv(parent_dir.as_posix()+'/data/processed/test_processed.csv',index=False)