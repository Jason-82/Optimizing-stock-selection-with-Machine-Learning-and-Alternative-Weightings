#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()


# In[2]:


import pandas as pd
import Risk_Kit2 as rk
import json
import logging
import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker import get_execution_role
import datetime



# In[3]:


sagemaker_session = sagemaker.Session()
role = get_execution_role()
endpoint_name='DEMO-deepar-2020-08-18-03-14-52-883'


# In[ ]:





# In[4]:


days=720
coin1='bitcoin'
coin2='ethereum'
vsCurrency='usd'
period='2y'
tickers='SPY AGG DIA XLY GLD SLV EFA VWO USO QQQ'
tickers_cryptos='BTC ETH'


# In[5]:


response=cg.get_coin_market_chart_by_id(coin1, vsCurrency, days)
response2=cg.get_coin_market_chart_by_id(coin2, vsCurrency, days)


# In[ ]:





# In[6]:


import yfinance as yf
tickersY = yf.Tickers(tickers)
df_hist=tickersY.history(period=period)
df_stocks=df_hist['Close']
df_stocks.columns=tickersY.symbols
df_stocks


# In[ ]:





# In[ ]:





# In[7]:


#tickersY.symbols


# In[8]:


df_btc=pd.DataFrame(data=response['prices'], columns=['Time', 'Price'])
df_btc


# In[9]:


df_eth=pd.DataFrame(data=response2['prices'], columns=['Time2', 'Price'])
df_eth


# In[10]:


df=pd.concat([df_btc, df_eth], axis=1)
df.drop(columns='Time2', inplace=True)


# In[11]:


df=df.pct_change()
df.drop(columns='Time', inplace=True)
#df.mean(axis=0)


# In[12]:



#df.drop(columns='Time', inplace=True)
df.columns=['BTC', 'ETH']
df.dropna(inplace=True)
df


# In[13]:


df_stocks=df_stocks.pct_change()
df_stocks.dropna(inplace=True)


# In[14]:


#rk.msr(0, df_rets_daily, dff.cov()['BTC'])


# In[15]:


'''window_size=33
start=0
end=start+window_size
df_mthly=pd.DataFrame()'''


# In[ ]:





# In[16]:


#df_stocks


# In[17]:


'''while end<2162:
    df_mthly=df_mthly.append(df.iloc[start:end].sum(axis=0), ignore_index=True)
    start+=window_size+1
    end=start+window_size'''


# In[18]:


#df_mthly=df_mthly.append(df_mthly.iloc[61:62].copy())


# In[19]:


#df_mthly


# In[20]:


#df_stocks


# In[21]:


'''the_diff=df_mthly.shape[0]-df_stocks.shape[0]
if the_diff<0:
    for i in range(abs(the_diff)):
        df_mthly=df_mthly.append(df_mthly.iloc[-1])
elif the_diff>0:
    for i in range(the_diff):
        df_stocks=df_stocks.append(df_stocks.iloc[-1])'''


# In[22]:


#df_mthly.index=df_stocks.index
#df_mthly


# In[23]:


#df_combined=pd.concat([df_stocks, df_mthly], axis=1)
#df_combined


# In[24]:


#daily_rets=df_combined.pct_change()
#daily_rets=df_combined.copy()


# In[25]:


'''daily_avg_rets_stocks=df_stocks.mean(axis=0)
daily_avg_rets_stocks'''


# In[26]:


'''daily_avg_rets_cryptos=df_mthly.mean(axis=0)
daily_avg_rets_cryptos'''


# In[27]:


#max_sharpe_weights_stocks=rk.msr(0, daily_avg_rets_stocks, df_stocks.cov())


# In[28]:


#max_sharpe_weights_stocks


# In[29]:


#max_sharpe_weights_cryptos=rk.msr(0, daily_avg_rets_cryptos, df_mthly.cov())
#max_sharpe_weights_cryptos


# In[30]:


#pd.DataFrame(max_sharpe_weights_stocks)


# In[31]:


#daily_rets.to_csv("daily-rets-crypto-and-stocks.csv")


# In[32]:


df_stocks.to_csv("daily-stock-returns.csv")


# In[33]:


s3 = boto3.resource(service_name='s3', region_name='us-west-2')
data = open('daily-stock-returns.csv', 'rb')
s3.Bucket('jason-test-2.0-sagemaker').put_object(Key='daily-stock-returns.csv', Body=data)


# In[34]:


df.to_csv("daily-crypto-returns.csv")


# In[35]:


s3 = boto3.resource(service_name='s3', region_name='us-west-2')
data = open('daily-crypto-returns.csv', 'rb')
s3.Bucket('jason-test-2.0-sagemaker').put_object(Key='daily-crypto-returns.csv', Body=data)


# In[ ]:





# In[36]:


freq = 'H'
prediction_length = 48
context_length = 72
t0 = '2016-01-01 00:00:00'
data_length = 400
num_ts = 200
period = 24


# In[37]:


def series_to_obj(ts, cat=None):
    print(ts.index)
    obj = {"start": str(ts.index[0]), "target": list(ts)}
    if cat is not None:
        obj["cat"] = cat
    return obj

def series_to_jsonline(ts, cat=None):
    return json.dumps(series_to_obj(ts, cat))


# In[ ]:





# In[38]:


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


# In[39]:


predictor = DeepARPredictor(
    endpoint_name='DEMO-deepar-2020-08-18-03-14-52-883',
    sagemaker_session=sagemaker_session,
    content_type="application/json"
)
predictor.set_prediction_parameters(freq, prediction_length)


# In[ ]:





# In[40]:


df_predictions_stocks=pd.DataFrame(index=tickers.split(), columns=['Returns'])


# In[41]:


for x,y in df_stocks.iteritems():
    temp=predictor.predict([y], content_type='application/json')
    df_predictions_stocks.loc[x,'Returns']=temp[0]['0.5'][0]
   


# In[42]:


#df_predictions_stocks


# In[ ]:





# In[ ]:





# In[43]:


for i in range(df.shape[0]):
    df.loc[i+1, "Date"]=datetime.datetime.now()-pd.to_timedelta(str(df.shape[0]-i-1)+"days")


# In[44]:


df.set_index(df['Date'], inplace=True)
df.drop(columns='Date', inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


df_predictions_cryptos=pd.DataFrame(index=tickers_cryptos.split(), columns=['Returns'])


# In[46]:


for x,y in df.iteritems():
    temp=predictor.predict([y], content_type='application/json')
    df_predictions_cryptos.loc[x,'Returns']=temp[0]['0.5'][0]


# In[47]:


df_predictions_cryptos


# In[48]:


df_predictions_stocks


# In[49]:


crypto_weights=rk.msr(0, df_predictions_cryptos, df.cov())
crypto_weights


# In[50]:


stock_weights=rk.msr(0, df_predictions_stocks, df_stocks.cov())
stock_weights


# In[51]:


df_predictions_stocks['Weights']=stock_weights


# In[52]:


df_predictions_cryptos['Weights']=crypto_weights


# In[53]:


df_predictions_stocks['Weights'].to_csv('Max-Sharpe-Weights-Stocks.csv')
df_predictions_cryptos['Weights'].to_csv('Max-Sharpe-Weights-Cryptos.csv')


# In[54]:


s3 = boto3.resource(service_name='s3', region_name='us-west-2')
data = open('Max-Sharpe-Weights-Stocks.csv', 'rb')
s3.Bucket('jason-test-2.0-sagemaker').put_object(Key='Max-Sharpe-Weights-Stocks.csv', Body=data)


# In[55]:


s3 = boto3.resource(service_name='s3', region_name='us-west-2')
data = open('Max-Sharpe-Weights-Cryptos.csv', 'rb')
s3.Bucket('jason-test-2.0-sagemaker').put_object(Key='Max-Sharpe-Weights-Cryptos.csv', Body=data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[84]:


pd.DataFrame(crypto_expected_return, index=['Returns', 'Weights'])


# In[56]:


stocks_expected_return=stock_weights@df_predictions_stocks*100


# In[57]:


crypto_expected_return=crypto_weights@df_predictions_cryptos*100


# In[96]:


pd.DataFrame(crypto_expected_return, index=['Returns', 'Weights']).to_csv("LSTM-Crypto-Daily-Return.csv")
pd.DataFrame(stocks_expected_return, index=['Returns', 'Weights']).to_csv("LSTM-Stocks-Daily-Return.csv")


# In[ ]:





# In[97]:


s3 = boto3.resource(service_name='s3', region_name='us-west-2')
data = open('LSTM-Crypto-Daily-Return.csv', 'rb')
s3.Bucket('jason-test-2.0-sagemaker').put_object(Key='LSTM-Crypto-Daily-Return.csv', Body=data)


# In[98]:


s3 = boto3.resource(service_name='s3', region_name='us-west-2')
data = open('LSTM-Stocks-Daily-Return.csv', 'rb')
s3.Bucket('jason-test-2.0-sagemaker').put_object(Key='LSTM-Stocks-Daily-Return.csv', Body=data)


# In[89]:


#total_return=crypto_expected_return*.1+stocks_expected_return*.9


# In[90]:


#total_return['Returns']


# In[91]:


#crypto_expected_return


# In[92]:


#PERCENTAGE_STOCKS=.9


# In[93]:


#total_return=df_stocks_LSTM*(PERCENTAGE_STOCKS)+df_cryptos_LSTM*(1-PERCENTAGE_STOCKS)


# In[110]:


#obj_cryptos = s4.get_object(Bucket='jason-test-2.0-sagemaker', Key='LSTM-Stocks-Daily-Return.csv')
#df_cryptos_LSTM_ = pd.read_csv(obj_cryptos['Body'])


# In[109]:


#s4 = boto3.client('s3')


# In[108]:


#df_cryptos_LSTM_


# In[107]:


#df_cryptos_LSTM_.loc[0][1]


# In[111]:


https://api.coinmarketcap.com/v1/ticker/ethereum/


# In[ ]:




