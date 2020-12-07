#!/usr/bin/env python
# coding: utf-8

# In[189]:


import pandas as pd
import boto3
import json
import yfinance as yf
s4 = boto3.resource(service_name='s3', region_name='us-west-2')


# In[190]:


table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp500 = table[0]
sp500s=sp500['Symbol'].tolist()


# In[191]:


#sp500s


# In[192]:


data = yf.download(sp500s,period="2y",interval="1D")
datan = data["Adj Close"].dropna(axis=1, how="all")  #dropna(axis=1,how="all")
datan = datan.dropna(axis=0, how="all")   #dropn


# In[193]:


#datan.head()


# In[194]:


#y=datan['AAPL'].pct_change().dropna()


# In[195]:


ENDPOINT_NAME = 'DEMO-deepar-2020-11-22-20-26-29-609'
runtime= boto3.client('runtime.sagemaker')
#df=pd.read_csv("https://s3-us-west-2.amazonaws.com/jason-test-2.0-sagemaker/stocks.csv")
    
'''df.index=df['Date']
df.drop(columns='Date', inplace=True)
df.index=pd.to_datetime(df.index)
        
df.dropna(inplace=True)
print(df)'''

finalList={}
for x,y in datan.iteritems():
    
    y=y.pct_change().dropna()
    spyList=y.tolist()
            
    payload={"start": str(datan.index[0]), "target": spyList}
   
    
    
    
    req={
        'instances': [payload],
        'configuration': {"num_samples": 1, "output_types": ["quantiles"], "quantiles": ["0.5"]}
        }
            
    req=json.dumps(req).encode('utf-8')
    resultList=1

    for z in range(21):

        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                           ContentType='application/json',
                                           Body=req)


        result = json.loads(response['Body'].read().decode())
        #print(result)
        my_prediction=result['predictions'][0]['quantiles']['0.5'][0]
        #print('Prediction\n')
        #print(my_prediction)

        resultList*=float(1+my_prediction)

        totalReturn=(resultList-1)
        finalList[y.name]=[totalReturn]
       
print(finalList)
df_final=pd.DataFrame(finalList)
df_final=df_final.T

df_final.columns=['21-day Predicted Return']
print(df_final)
df_final=df_final.sort_values(by='21-day Predicted Return', ascending=False)


# In[ ]:





# In[196]:


df_final.iloc[:100].to_csv('Top 30 21-day Returns by LSTM.csv')


# In[197]:


#df_final=df_final.sort_values(by='21-day Predicted Return', ascending=False)


# In[198]:


#df_final


# In[199]:


data = open('Top 30 21-day Returns by LSTM.csv', 'rb')
s4.Bucket('lstm-returns-to-be-pulled-by-facebook-prophet').put_object(Key='Top 30 21-day Returns by LSTM.csv', Body=data)
        


# In[200]:


'''s3 = boto3.client('s3')
obj = s3.get_object(Bucket="lstm-returns-to-be-pulled-by-facebook-prophet", Key="Top 30 21-day Returns by LSTM.csv")
df = pd.read_csv(obj['Body'])
df'''


# In[207]:


percent=5
d1={'AGG': percent+'%'}


# In[206]:


d1


# In[203]:


d1.update({'AGG': {percent: 60}})


# In[204]:


d1


# In[208]:


obj_weighting_styles = s3.get_object(Bucket='future-weights-and-backtested-returns-for-each-weighting-style', Key='Final_Future_Weights.csv')
df_weightings = pd.read_csv(obj_weighting_styles['Body'])
df_weightings.set_index('Unnamed: 0', inplace=True)


# In[210]:


df_weightings


# In[211]:


stockpercent=df_weightings[df_weightings['Stock']=='HST']['MSR']


# In[218]:


stockpercent[stockpercent.index[0]]


# In[219]:


obj_stocks = s3.get_object(Bucket='facebook-prophet-layer', Key='Prophet_Returns.csv')
df_returns = pd.read_csv(obj_stocks['Body'])


# In[227]:


df2=df_returns[df_returns['Decision']=='buy']


# In[251]:


obj_stocks = s3.get_object(Bucket='facebook-prophet-layer', Key='Prophet_Covariance.csv')
df_cov = pd.read_csv(obj_stocks['Body'])


# In[291]:


#df_cov


# In[292]:


#df_returns['Return']


# In[293]:


#df_cov['Unnamed: 0']==df2.reset_index()['Stock']


# In[294]:


#df2['Stock'].tolist()


# In[295]:


#df_cov['Unnamed: 0'][0] in df2['Stock'].tolist()


# In[296]:


#dff=pd.DataFrame()


# In[297]:


'''for i in range(df_cov.shape[0]):
    if df_cov['Unnamed: 0'][i] in df2['Stock'].tolist():
        dff=dff.append(pd.DataFrame({df_cov['Unnamed: 0'][i]: df_cov.iloc[i]}).T)'''


# In[298]:


#dff[df2['Stock'].tolist()].shape


# In[299]:


#pd.DataFrame({df_cov['Unnamed: 0'][i]: df_cov.iloc[i]}).T


# In[300]:


#dff=dff.append(pd.DataFrame({df_cov['Unnamed: 0'][i]: df_cov.iloc[i]}).T)


# In[301]:


#dff


# In[ ]:




