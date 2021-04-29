#!/usr/bin/env python
# coding: utf-8

# # Libraries and Installations

# In[ ]:


get_ipython().system('pip install sagemaker-experiments')
get_ipython().system('pip install s3fs')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install shap')
get_ipython().system('pip install smdebug')


# In[ ]:


from io import StringIO
import numpy as np
import os
import pandas as pd
import boto3
import time
import s3fs
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import re
#import shap
from scipy import stats
import copy


# In[ ]:


import sagemaker
from sagemaker import get_execution_role
from sagemaker.analytics import ExperimentAnalytics

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.debugger import rule_configs, Rule, DebuggerHookConfig,CollectionConfig
from sagemaker.estimator import Estimator
from sagemaker.session import s3_input
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import Session

from smdebug.trials  import create_trial


# # Configs

# In[ ]:


get_ipython().system('pip install awscli')


# In[ ]:


import boto3
session = boto3.Session(
    aws_access_key_id='AKIA26LUPUCZ7UWUXJHT',
    aws_secret_access_key='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',
)


# In[ ]:


now = datetime.now()

current_time = now.strftime("%Y-%m-%d--%H-%M-%S")
print("current_time:",current_time)

sagemaker_session = sagemaker.Session()

# Change "get_execution_role()" to access role name for local deployment
role = 'arn:aws:iam::752400441523:role/Sagemaker_Access'

bucket = '1905-assignment2-sm'
prefix = 'Scikit-pre-model-Inference-Pipelines'

train_data = 's3://1905-assignment2-sm/housing/imput-datasets/train_data_without_header.csv'
test_data = 's3://1905-assignment2-sm/housing/imput-datasets/test_data_without_header.csv'

FRAMEWORK_VERSION = "0.23-1"
script_path = 'sklearn_pipeline.py'
dependency_path ='dependencies.py'

base_job_name = f"Builtin-XGB-algo-{current_time}"

output_data_prefix = f'housing/datasets/output/{base_job_name}'
data_output_path = f's3://{bucket}/{output_data_prefix}'

debug_prefix = f'housing/jobs/debug/{base_job_name}'
debug_path = f's3://{bucket}/{debug_prefix}'

experiment_name_prefix = "builtin-xgboost-track13"


# In[ ]:


print(train_data)


# In[ ]:


role


# # Batch transform

# ## Fit the train data

# In[ ]:


from sagemaker.local import LocalSession


# In[ ]:


sklearn_preprocessor = SKLearn(
    entry_point = script_path,
    role = role,
    framework_version = FRAMEWORK_VERSION,
    train_instance_type = 'local', # "ml.m5.xlarge", #"local" ,
    train_use_spot_instance = True,
    train_max_run = 600,
    train_max_wait = 1200,
    dependencies = [dependency_path],
    #sagemaker_session = sagemaker_session
)


# In[ ]:


role


# In[ ]:


check = pd.read_csv(train_data)
check.head()


# In[ ]:


sklearn_preprocessor.fit(
    inputs={'train':train_data},
#    job_name=base_job_name
)


# ## Transform the training data

# In[ ]:


transformer = sklearn_preprocessor.transformer(
    instance_count=1,
    instance_type='local',#'ml.m5.xlarge',
    assemble_with = 'Line',
    accept = 'text/csv',
    output_path=data_output_path)


# In[ ]:


transformer.transform(
    data=train_data,
    content_type="text/csv",
    job_name=base_job_name+'-train')

print("Waiting for transform job:" + transformer.latest_transform_job.job_name)
transformer.wait()


# In[ ]:


preprocessed_train_data = transformer.output_path


# In[ ]:


preprocessed_train_data


# ## Transform the test data

# In[ ]:


transformer.transform(
    data=test_data,
    content_type="text/csv",
    job_name=base_job_name+"-test")

print("Waiting for transform job:" + transformer.latest_transform_job.job_name)
transformer.wait()


# In[ ]:


preprocessed_test_data = transformer.output_path


# In[ ]:


f'{output_data_prefix}'


# ## Upload processed data to s3

# In[ ]:


client = boto3.client('s3')
obj = client.get_object(Bucket=bucket, Key = f'{output_data_prefix}/train_data_without_header.csv.out')
body = obj['Body']
csv_string = body.read().decode('utf-8')
processed_train_data = pd.read_csv(StringIO(csv_string))


# In[ ]:


train_file = 'processed_train_data.csv'
processed_train_data.to_csv(train_file,index=False,header=False)
with open(train_file,'rb') as data:
    boto3.Session().resource('s3').Bucket(bucket).upload_fileobj(data,os.path.join(output_data_prefix,'processed-train-data.csv'))


# In[ ]:


obj = client.get_object(Bucket=bucket, Key = f'{output_data_prefix}/test_data_without_header.csv.out')
body = obj['Body']
csv_string = body.read().decode('utf-8')
processed_test_data = pd.read_csv(StringIO(csv_string))

test_file = 'processed_test_data.csv'
processed_test_data.to_csv(test_file,index=False,header=False)
with open(test_file,'rb') as data:
    boto3.Session().resource('s3').Bucket(bucket).upload_fileobj(data,os.path.join(output_data_prefix,'processed-test-data.csv'))


# ## Real time Prediction using endpoint

# In[ ]:


from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
import boto3
from time import gmtime, strftime
from sagemaker.estimator import Estimator
from sagemaker import PipelineModel

timestamp_prefix = current_time

scikit_learn_inferencee_model = sklearn_preprocessor.create_model()
scikit_learn_inferencee_model.env = {"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT":"text/csv"}
model_containers = [scikit_learn_inferencee_model]

model_name = 'inference-pipeline-' + current_time
endpoint_name = 'inference-pipeline-ep-' + current_time

sm_model = PipelineModel(
            name=model_name,
            role=role,
            models=model_containers)

predictor = sm_model.deploy(initial_instance_count=1,
                           instance_type='local',#'ml.m5.xlarge',
                           endpoint_name=endpoint_name,
                           #data_capture_config=data_capture_config
                           )

from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

payload = "-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,NEAR BAY"

predictor = Predictor(
        endpoint_name = endpoint_name,
        sagemaker_session = sagemaker_session,
        serializer = CSVSerializer(),
        deserializer = JSONDeserializer(),
    )


# In[ ]:


print(predictor.predict(data=payload))


# In[ ]:


#Delete the endpoint
#sm_client = sagemaker_session.boto_session.client('sagemaker')
#sm_client.delete_endpoint(EndpointName=endpoint_name)

