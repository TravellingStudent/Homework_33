# <YOUR_IMPORTS>
import os
import dill
import json
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def predict():
    # <YOUR_CODE>
    path = os.environ.get('PROJECT_PATH', '.')
    dir_model_name =  f'{path}/data/models/'
    model_filename = os.listdir(path = dir_model_name)
    #print(dir_model_name+model_filename[-1])
    with open(dir_model_name+model_filename[-1],'rb') as input_file:
        model = dill.load(input_file)

    dir_data_name = f'{path}/data/test/'
    dir_predict_name = f'{path}/data/predictions/'

    data_filenames = os.listdir(path=dir_data_name)
    i=0
    for name in data_filenames:
        with open(dir_data_name+name) as fp:
            js_data = json.load(fp)
        df=pd.json_normalize(js_data)
        y = model.predict(df)
        df['ṕred'] = y
        if i == 0:
            Data = df[['id','ṕred']]
            i += 1
        else:
            Data = pd.concat([Data, df[['id','ṕred']]])

    Data.to_csv(f'{dir_predict_name}predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
