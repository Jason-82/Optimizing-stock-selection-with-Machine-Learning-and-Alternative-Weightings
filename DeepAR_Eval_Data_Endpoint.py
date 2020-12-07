#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
np.random.seed(1)
import pandas as pd
import json
import matplotlib.pyplot as plt
import boto3
import s3fs
import sagemaker
from sagemaker import get_execution_role


# In[2]:


prefix = 'sagemaker/DEMO-deepar2'

sagemaker_session = sagemaker.Session()
role = get_execution_role()
bucket = 'jason-test-2.0-sagemaker'
return_file='daily-stock-returns.csv'
s3_data_path = "{}/{}/data".format(bucket, prefix)
s3_output_path = "{}/{}/output".format(bucket, prefix)


# In[3]:


from sagemaker.amazon.amazon_estimator import get_image_uri
image_name = get_image_uri(boto3.Session().region_name, 'forecasting-deepar')


# In[4]:


freq = 'H'
prediction_length = 48
context_length = 72
t0 = '2016-01-01 00:00:00'
data_length = 400
num_ts = 200
period = 24


# In[ ]:





# In[5]:


s3 = boto3.client('s3')
obj = s3.get_object(Bucket=bucket, Key=return_file)
df = pd.read_csv(obj['Body'])
df


# In[6]:


df.set_index('Date', inplace=True)
df


# In[7]:


df.index=pd.to_datetime(df.index)


# In[ ]:





# In[8]:


spy=df['SPY']
spy=spy.pct_change()
spyList=spy.tolist()
#spyList.pop(0)
#spyList
spy=spy.replace([np.inf, -np.inf], np.nan)
spy.dropna(inplace=True)


# In[48]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


def series_to_obj(ts, cat=None):
    print(ts.index)
    obj = {"start": str(ts.index[0]), "target": list(ts)}
    if cat is not None:
        obj["cat"] = cat
    return obj

def series_to_jsonline(ts, cat=None):
    return json.dumps(series_to_obj(ts, cat))


# In[10]:


encoding = "utf-8"
s3filesystem = s3fs.S3FileSystem()

with s3filesystem.open(s3_data_path + "/train/train.json", 'wb') as fp:
   
    fp.write(series_to_jsonline(spy).encode(encoding))
    fp.write('\n'.encode(encoding))

with s3filesystem.open(s3_data_path + "/test/test.json", 'wb') as fp:
   
    fp.write(series_to_jsonline(spy).encode(encoding))
    fp.write('\n'.encode(encoding))


# In[10]:


estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sagemaker_session,
    image_uri=image_name,
    role=role,
    train_instance_count=1,
    train_instance_type='ml.c4.xlarge',
    base_job_name='DEMO-deepar',
    output_path="s3://" + s3_output_path
)


# In[11]:


hyperparameters = {
    "time_freq": freq,
    "context_length": str(context_length),
    "prediction_length": str(prediction_length),
    "num_cells": "40",
    "num_layers": "3",
    "likelihood": "gaussian",
    "epochs": "20",
    "mini_batch_size": "32",
    "learning_rate": "0.001",
    "dropout_rate": "0.05",
    "early_stopping_patience": "10"
}


# In[12]:


estimator.set_hyperparameters(**hyperparameters)


# In[13]:


data_channels = {
    "train": "s3://{}/train/".format(s3_data_path),
    "test": "s3://{}/test/".format(s3_data_path)
}

estimator.fit(inputs=data_channels)


# In[11]:


'''payload={"start": "2016-01-01 00:00:00", "target": spyList}
req={
        'instances': [payload],
                'configuration': {"num_samples": 100, "output_types": ["quantiles"], "quantiles": ["0.5"]}
            }'''


# In[12]:


#req=json.dumps(req).encode('utf-8')


# In[19]:


job_name = estimator.latest_training_job.name

endpoint_name = sagemaker_session.endpoint_from_job(
    job_name=job_name,
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
    image_uri=image_name,
    role=role
)


# In[28]:


class DeepARPredictor(sagemaker.predictor.RealTimePredictor):

    def set_prediction_parameters(self, freq, prediction_length):
        """Set the time frequency and prediction length parameters. This method **must** be called
        before being able to use `predict`.        
        Parameters:
        freq -- string indicating the time frequency
        prediction_length -- integer, number of predicted time points
        
        Return value: none.
        """
        self.freq = freq
        self.prediction_length = prediction_length
        
    def predict(self, ts, cat=None, encoding="utf-8", num_samples=100, quantiles=["0.1", "0.5", "0.9"], content_type="application/json"):
        """Requests the prediction of for the time series listed in `ts`, each with the (optional)
        corresponding category listed in `cat`.
        
        Parameters:
        ts -- list of `pandas.Series` objects, the time series to predict
        cat -- list of integers (default: None)
        encoding -- string, encoding to use for the request (default: "utf-8")
        num_samples -- integer, number of samples to compute at prediction time (default: 100)
        quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])
        
        Return value: list of `pandas.DataFrame` objects, each containing the predictions
        """
        
        prediction_times=[]
        req=[]
        if type(ts)==list:
            prediction_times = [x.index[-1]+pd.Timedelta(1, unit=self.freq) for x in ts]
            req = self.__encode_request(ts, cat, encoding, num_samples, quantiles)
        elif type(ts)==dict:
            
            prediction_times=[]
            target_len=len(ts['target'])
            t0=ts['start']
            prediction_times.append(t0)
            
                
                
            req={
                'instances': [ts],
                'configuration': {"num_samples": 100, "output_types": ["quantiles"], "quantiles": ["0.5"]}#["0.1", "0.5", "0.9"]}
            }
            req=json.dumps(req).encode('utf-8')
        elif type(ts)==bytes:
            prediction_times=[]
            req=ts
            print("IN HERE")
            prediction_times.append(json.loads(req)['instances'][0]['start'])
        
        res = super(DeepARPredictor, self).predict(req, initial_args={"ContentType": content_type})
        
        return self.__decode_response(res, prediction_times, encoding)
    
    def __encode_request(self, ts, cat, encoding, num_samples, quantiles):
        
        instances = [series_to_obj(ts[k], cat[k] if cat else None) for k in range(len(ts))]
        configuration = {"num_samples": num_samples, "output_types": ["quantiles"], "quantiles": quantiles}
        http_request_data = {"instances": instances, "configuration": configuration}
        return json.dumps(http_request_data).encode(encoding)
    
    def __decode_response(self, response, prediction_times, encoding):
        response_data = json.loads(response.decode(encoding))
        list_of_df = []
        
        for k in range(len(prediction_times)):
            
            prediction_index = pd.date_range(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            list_of_df.append(pd.DataFrame(data=response_data['predictions'][k]['quantiles'], index=prediction_index))
       
        return list_of_df


# In[29]:


predictor = DeepARPredictor(
    endpoint_name='DEMO-deepar-2020-08-18-03-14-52-883',
    sagemaker_session=sagemaker_session,
    content_type="application/json"
)
predictor.set_prediction_parameters(freq, prediction_length)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


dfts = predictor.predict([spy], content_type='application/json')


# In[27]:


dfts[0]['0.5']


# In[ ]:




