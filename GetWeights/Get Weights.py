import pandas as pd
import json
import boto3
import Risk_Kit2 as rk
import yfinance as yf
import json
s3 = boto3.client('s3')
s4 = boto3.resource(service_name='s3', region_name='us-west-2')



def lambda_handler(event, context):
    
    final_future_weights, df_all_backtests_returns=getWeightsReturns()
    #return final_future_weights, df_all_backtests_returns
    final_future_weights.to_csv("/tmp/Final_Future_Weights.csv")
    data = open('/tmp/Final_Future_Weights.csv', 'rb')
    s4.Bucket('future-weights-and-backtested-returns-for-each-weighting-style').put_object(Key='Final_Future_Weights.csv', Body=data)
    
    df_all_backtests_returns.to_csv("/tmp/All_Backtest_Returns.csv")
    data = open('/tmp/All_Backtest_Returns.csv', 'rb')
    s4.Bucket('future-weights-and-backtested-returns-for-each-weighting-style').put_object(Key='All_Backtest_Returns.csv', Body=data)
    
    message=f'{json.loads(final_future_weights.to_json())} and {json.loads(df_all_backtests_returns.to_json())}'
    return {
        "sessionAttributes": {
          "SuggestedFutureWeights": json.loads(final_future_weights.to_json()),
          "BacktestedReturns": json.loads(df_all_backtests_returns.to_json())
        },
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": 'Fulfilled',
            "message": {
      "contentType": "PlainText",
      "content": message
    }}}
    
def getWeightsReturns():
    
    obj_stocks = s3.get_object(Bucket='facebook-prophet-layer', Key='Prophet_Returns.csv')
    df_returns = pd.read_csv(obj_stocks['Body'])
    obj_stocks = s3.get_object(Bucket='facebook-prophet-layer', Key='Prophet_Covariance.csv')
    df_cov = pd.read_csv(obj_stocks['Body'])
    #df_cov.set_index('Unnamed: 0', inplace=True)
    
    
    df_returns=df_returns[df_returns['Decision']=='buy']
    df_covariance=pd.DataFrame()
    for i in range(df_cov.shape[0]):
        if df_cov['Unnamed: 0'][i] in df_returns['Stock'].tolist():
            df_covariance=df_covariance.append(pd.DataFrame({df_cov['Unnamed: 0'][i]: df_cov.iloc[i]}).T)
    df_covariance=df_covariance[df_returns['Stock'].tolist()]
    #df_cov=df_cov[df_returns['Stock'].tolist()]
    #return df_returns['Return'].shape, df_covariance.shape
    weights_msr=rk.msr(0,df_returns['Return'], df_covariance)
    equal_risk_weights=rk.equal_risk_contributions(df_covariance)
    weights_gmv=rk.gmv(df_covariance)
    tickers=list(df_returns['Stock'])
    
    tickersY = yf.Tickers(tickers)
    df_hist=tickersY.history(period='2y')
    df_stocks=df_hist['Close']
    df_stocks.columns=tickersY.symbols
    df_stocks=df_stocks.pct_change()
    
    backtest_msr, returns_msr=rk.backtest_ws(df_stocks, 60, weighting=rk.getMSRWeights)
    backtest_ew, returns_ew=rk.backtest_ws(df_stocks, 60)
    print("WHAT")
    equal_risk_weights_backtest, returns_equal_risk=rk.backtest_ws(df_stocks, 60, weighting=rk.equal_risk_contributions_Helper)
    gmv_weights, returns_gmv=rk.backtest_ws(df_stocks, 60, weighting=rk.weight_gmv)
      
    print("Hello")
    future_weights=pd.DataFrame({'Stock': tickers,
                                 'MSR': weights_msr,
                                'EQ Risk': equal_risk_weights,
                                'GMV': weights_gmv})
                                
    rounded_future_weights=round(future_weights[['MSR', 'EQ Risk', 'GMV']]*100,2)
    final_future_weights=pd.concat([future_weights['Stock'], rounded_future_weights], axis=1)
      
      
    df_all_backtests=pd.DataFrame({'GMV': returns_gmv,
                                 'MSR': returns_msr,
                                 'Equal Weight': returns_ew,
                                 'Equal Risk': returns_equal_risk})
                                 
    df_all_backtests_returns=round(((1+df_all_backtests).prod()-1)*100,2)
      
    return final_future_weights, df_all_backtests_returns