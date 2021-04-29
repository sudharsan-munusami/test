#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil


import argparse
import csv
import json
import joblib as joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

from dependencies import CombinedAttributesAdder

from sagemaker_containers.beta.framework import (content_types,encoders,env,modules, transformer,worker)

feature_columns_names = [
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income','ocean_proximity']

label_column ='median_house_value'

feature_columns_dtype = {
    'ocean_proximity': "category",
    'longitude' : "float64",
    'latitude': "float64",
    'housing_median_age':"float64",
    'total_rooms':"float64",
    'total_bedrooms':"float64",
    'population':"float64",
    'median_income':"float64",
    'households':"float64"
}

label_column_dtype = {'median_house_value':"float64"}

def merge_two_dicts(x,y):
    z = x.copy()
    z.update(y)
    return z

if __name__ == '__main__':
    print("*********************")
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir',type=str,default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir',type=str,default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train',type=str,default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    input_files = [os.path.join(args.train,file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}. \n' +
                         'This usually indicates that the channel ({}) was incorrectly pecified,\n'+
                         'The data specification in s3 was incorrectly specified or the role specified/n'+
                         'does not have permission to access the data').format(args.train,"train"))
    
    raw_data = [pd.read_csv(
                file, header =None,
                names=feature_columns_names + [label_column],
                dtype = merge_two_dicts(feature_columns_dtype, label_column_dtype)) for file in input_files]
    concat_data = pd.concat(raw_data)
    concat_data.head()
    
    concat_data.drop(label_column, axis=1,inplace=True)
    
    print("*****Executing numeric transformer******")
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        CombinedAttributesAdder(),
        StandardScaler())
    
    print("*****Executing categorical transformer*****")
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant',fill_value='missing'),
    OneHotEncoder(handle_unknown='ignore'))
    
    print("*****Executing Column Transformer*****")
    preprocessor = ColumnTransformer(transformers=[
        ("num",numeric_transformer,make_column_selector(dtype_exclude="category")),
        ("cat",categorical_transformer,make_column_selector(dtype_include="category"))])
    print("preprocessor")
    print(preprocessor)
    
    print("*****Executing FIT*****")
    print(concat_data.columns)
    preprocessor.fit(concat_data)
    
    joblib.dump(preprocessor, os.path.join(args.model_dir,"model.joblib"))
    
    print("saved model!")
    
def input_fn(input_data,content_type):
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data),header=None)
        if len(df.columns) == len(feature_columns_names) + 1:
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            df.columns = feature_columns_names
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        
def output_fn(prediction,accept):
    print("*****output function*****")
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features":row})
        json_output = {"instances":instances}
        return worker.Response(json.dumps(json_output),mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction,accept),mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
        
def predict_fn(input_data,model):
    print("*****predict function*****")
    features = model.transform(input_data)
    if label_column in input_data:
        return np.insert(features,0,input_data[label_column],axis=1)
    else:
        return features

def model_fn(model_dir):
    print("*****model function*****")
    print(model_dir)
    preprocessor = joblib.load(os.path.join(model_dir,"model.joblib"))
    return preprocessor

