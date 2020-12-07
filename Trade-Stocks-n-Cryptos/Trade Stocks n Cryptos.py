from fbprophet import Prophet
import alpaca_trade_api as tradeapi
import yfinance as yf
import numpy as np
import pandas as pd
#import Risk_Kit2 as rk
import boto3
import json
import datetime as dt
import os
pd.plotting.register_matplotlib_converters()  # https://github.com/facebook/prophet/issues/999
s3 = boto3.client('s3')
s4 = boto3.resource(service_name='s3', region_name='us-west-2')
ALPACAUSER =str(os.environ['ALPACAUSER'])
ALPACAPW = str(os.environ['ALPACAPW'])
api = tradeapi.REST(
        # TRADINGBOT1
        ALPACAUSER,
        ALPACAPW,
        'https://paper-api.alpaca.markets'
    )

weightingTypes={
  'Maximum Sharpe Ratio': 'MSR',
  'Global Minimum Variance': 'GMV',
  'Equal Weight': 'EW',
  'Equal Risk Contribution': 'EQ Risk'
}

def ElicitSlot(df_backtest_returns, event, slotToGet):
  
  weightingStyle=event['currentIntent']['slots']['WeightingStyle']
  trade=event['currentIntent']['slots']['TRADE']
  message=json.loads(df_backtest_returns['0'].to_json())
  
  if slotToGet=='WeightingStyle':
    message+=f'Choose your weighting style'
  else:
    message=f'Do you want to execute the trades?'
  return {
     "dialogAction": {
      
    "type": "ElicitSlot",
    "message": {
      "contentType": "PlainText",
      "content": message
    },
   "intentName": "Get_Recommended_Buys_Backtests",
   "slots": {
      "WeightingStyle": weightingStyle,
      "TRADE": trade
   },
   "slotToElicit" : slotToGet,
}
}



# alpaca functions
def buy(api,stock,howmany,currentprice,investments):
  print("IN BUY")
  print(stock)
  try:
    api.submit_order(
      symbol=stock,
      qty=howmany,
      side='buy',
      type='market',
      time_in_force='gtc',
      order_class='bracket',
      stop_loss={'stop_price': currentprice * 0.98,
                'limit_price':  currentprice * 0.97},
      take_profit={'limit_price': currentprice * 1.4}
    )
    
    investments.update({stock:{investments[stock]:f'${round(howmany*currentprice,2)}'}})
    return investments
  except Exception as e:
    print("Error when trying to buy ",stock)
    print(e)

def sell(api,stock,howmany,investments):
  try:
    api.submit_order(
      symbol=stock,
      qty=howmany,
      side='sell',
      type='market',
      time_in_force='gtc'
    )
    investments.update({stock:-howmany})
    return investments
  except Exception as e:
    print("Error when trying to sell ",stock)
    print(e)

# PARAMETERS & CONTROL

def run(event, ALPACAUSER,ALPACAPW, TRADE=True, weightingStyle="Equal Weight"):
  '''obj_1day_returns = s3.get_object(Bucket='facebook-prophet-layer', Key='1-day-Prophet-Returns.csv')
  df_full_returns = pd.read_csv(obj_1day_returns['Body'])'''
  
  obj_summary_rets = s3.get_object(Bucket='facebook-prophet-layer', Key='Prophet_Returns.csv')
  df_summary_rets = pd.read_csv(obj_summary_rets['Body'])
  
  obj_1day_prices = s3.get_object(Bucket='facebook-prophet-layer', Key='Prophet-Full-Prices.csv')
  df_full_prices = pd.read_csv(obj_1day_prices['Body'])
  
  obj_weighting_styles = s3.get_object(Bucket='future-weights-and-backtested-returns-for-each-weighting-style', Key='Final_Future_Weights.csv')
  df_weightings = pd.read_csv(obj_weighting_styles['Body'])
  df_weightings.set_index('Unnamed: 0', inplace=True)
  
  #obj_backtest_returns = s3.get_object(Bucket='facebook-prophet-layer', Key='All_Backtest_Returns.csv')
  #df_backtest_returns = pd.read_csv(obj_backtest_returns['Body'])
  weightingStyle=event['currentIntent']['slots']['WeightingStyle']
  
  
  
  desiredInvestmentAmount=event['sessionAttributes']['DesiredInvestmentAmount']
  
  account = api.get_account()
  buyingpower = float(account.buying_power) * 0.95 # to avoid overspending

  
  
  decisions=df_summary_rets.set_index('Stock')['Decision'].to_dict()
  nrbuys = len(list(decisions.values())[list(decisions.values())=="buy"])
  
  if weightingStyle=='Global Minimum Variance':
    weight_Abbr='GMV'
  elif weightingStyle=='Equal Risk Contribution':
    weight_Abbr='EQ Risk'
  elif weightingStyle=='Maximum Sharpe Ratio':
    weight_Abbr='MSR'
  else:
    weight_Abbr=None
  
  
          # final buy
  investments={}
  if TRADE:
    print("TRADE IS A YES")
    buyingpower = float(account.buying_power) * 0.95
    buyingpower=min(buyingpower, float(desiredInvestmentAmount))
    for stock in decisions.keys():
      #print("STOCK")
      #print(stock)
      
      dec = decisions[stock]
      #print("DEC")
      #print(dec)
      if dec == "buy":
        #print(df_full_prices[stock])
        if not weight_Abbr is None:
          stockWeightPercent=df_weightings[df_weightings['Stock']==stock][weight_Abbr]
          stockWeightPercent=stockWeightPercent[stockWeightPercent.index[0]]
          amount = (1/100*stockWeightPercent*buyingpower)
          #print(df_weightings)
          #print(stock)
        else:
          amount=float(buyingpower) / nrbuys # equal weights
        #print("BUYING POWER AND AMOUNT")
        #print(buyingpower)
        #print(amount)
        currentprice = df_full_prices[stock].iloc[-1]
        #print("CURRENT PRICE")
        #print(currentprice)
        howmany = int(amount / currentprice)
        if howmany > 0:
          print("HOWMANY")
          print(amount)
          print(stockWeightPercent)
          investments.update({stock: f'{stockWeightPercent}%'})
          print("between")
          print(investments)
          investments = buy(api,stock,howmany,currentprice,investments)
          print("After call")
          print(investments)
          print("Buying %d of %s"%(amount,stock))
      elif dec == "sell": # sell if holded
        holdingStock = int(api.get_position(stock).qty)
        investments = sell(api,stock,holdingStock,investments)
        print("Selling %d of %s"%(holdingStock,stock))
      else:
        print("No decision for ",stock)
    return investments, decisions
  else:
    return f'Did not trade', ""


def lambda_handler(event, context):
    
    #if event['currentIntent']['name']=='MakeOptimalPortfolioTrades':
    
    weightingStyle=event['currentIntent']['slots']['WeightingStyle']
    trade=event['currentIntent']['slots']['TRADE']
    source = event["invocationSource"]
    if source == "DialogCodeHook":
      obj_backtest_returns = s3.get_object(Bucket='future-weights-and-backtested-returns-for-each-weighting-style', Key='All_Backtest_Returns.csv')
      df_backtest_returns = pd.read_csv(obj_backtest_returns['Body'])
      df_backtest_returns=df_backtest_returns.set_index('Unnamed: 0')
        
      
        #else: return Delegate(event, weightingStyle)
         
      if not weightingStyle:
      
        return ElicitSlot(df_backtest_returns, event, "WeightingStyle")
          
      if not trade:
        return ElicitSlot(df_backtest_returns, event, "TRADE")
        
      
          
          
      weightingStyle=event['currentIntent']['slots']['WeightingStyle']
      trade=event['currentIntent']['slots']['TRADE']    
      
      return delegate(event['sessionAttributes'], event['currentIntent']['slots'])
   
    else:
      if event['currentIntent']['slots']['TRADE']=='yes':
        trade=True
      else:
        trade=False
      weightingStyle=event['currentIntent']['slots']['WeightingStyle']
        
    
      try:
          GETNEWCONSTITUENTS =True 
          #bool(os.getenv("C"))
          #(str(os.environ['GETNEWCONSTITUENTS']))
          ALPACAUSER =str(os.environ['ALPACAUSER'])
          ALPACAPW = str(os.environ['ALPACAPW'])
          investments, decisions = run(event, ALPACAUSER,ALPACAPW,trade,weightingStyle)
          print("BACK IN MAIN")
          print(investments) # INvestments = final positions
          print(type(investments))
          
          message=f'Investments: {investments}'
          return {
          "sessionAttributes": {
            "Investments": str(investments),
            "Decisions": str(decisions)
          },
          "dialogAction": {
              "type": "Close",
              "fulfillmentState": 'Fulfilled',
              "message": {
        "contentType": "PlainText",
        "content": message
      }}}
      except Exception as e:
          raise # for debugging
          return {
              'statusCode': 400,
              'body': e}
              
'''def delegate(event, weightingStyle, trade):
  return {
    "dialogAction": {
   "type": "Delegate",
   "slots": {
      "WeightingStyle": weightingStyle,
      "TRADE": trade
   },
   }
}
   '''         
def delegate(session_attributes, slots, type="Delegate", intent=None):

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {"type": type, "slots": slots},
    }
