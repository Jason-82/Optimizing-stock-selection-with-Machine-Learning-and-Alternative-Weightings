import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import alpaca_trade_api as tradeapi
import boto3
import os
import urllib3
ETHPLORER_API="EK-4LskG-ZW9WYSb-dAAu3"
PERSONAL_BANK_ACCOUNT="0x3c8AF5Eb646b637E2585994e400a7d32F1D58C0b"
http = urllib3.PoolManager()
ETHERSCAN_API="Y6HRJC1BWBDJF234U51USVPGGGBPEM2T8W"
ALPACAUSER =str(os.environ['ALPACAUSER'])
ALPACAPW = str(os.environ['ALPACAPW'])
api = tradeapi.REST(
        # TRADINGBOT1
        ALPACAUSER,
        ALPACAPW,
        'https://paper-api.alpaca.markets'
    )


s3 = boto3.client('s3')




def lambda_handler(event, context):
   
    desiredInvestmentAmount=event['currentIntent']['slots']['DesiredInvestmentAmount']

    event['sessionAttributes']['DesiredInvestmentAmount']=desiredInvestmentAmount
    source = event["invocationSource"]
    if source == "DialogCodeHook":
        return delegate(event['sessionAttributes'], event['currentIntent']['slots'])
    else:
        url="https://api-kovan.etherscan.io/api?module=account&action=balance&address="+PERSONAL_BANK_ACCOUNT+"&tag=latest&apikey="+ETHERSCAN_API
        r = http.request('GET', url)
        event['sessionAttributes']['BankAccountBalance']=json.loads(r.data.decode('utf-8'))['result']
        
            
        
        account = api.get_account()
        buyingpower = float(account.buying_power)
        event['sessionAttributes']['InvestmentAccountBalance']=buyingpower
        #desiredInvestmentAmount=min(float(desiredInvestmentAmount), buyingpower)
        bankAccountBalance=event['sessionAttributes']['BankAccountBalance']
        investmentAccountBalance=event['sessionAttributes']['InvestmentAccountBalance']
        if float(investmentAccountBalance)>=float(desiredInvestmentAmount):
            message=f"You've got ${investmentAccountBalance} in your investment account, you're ready to trade. Want to see your optimal portfolio?"
        else:
            message=f"You don't have enough in your investment account...you're short ${float(desiredInvestmentAmount)-investmentAccountBalance} "
            
            withdrawalNeeded=float(desiredInvestmentAmount)-float(investmentAccountBalance)
            if float(bankAccountBalance)>=withdrawalNeeded:
                message+=f"You've got enough to make up the difference in your bank account. Want to make a transfer by withdrawing from your bank?"
            else:
                message+=f"Sorry, you just don't have enough money"
            
        return {
            "sessionAttributes": {
              "BankAccountBalance": bankAccountBalance,
              "InvestmentAccountBalance": investmentAccountBalance,
              "DesiredInvestmentAmount": desiredInvestmentAmount,
              "WithdrawalNeeded": withdrawalNeeded
            },
            "dialogAction": {
                "type": "Close",
                "fulfillmentState": 'Fulfilled',
                "message": {
          "contentType": "PlainText",
          "content": message
        }}}
    
    
    
def delegate(session_attributes, slots, type="Delegate", intent=None):

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {"type": type, "slots": slots},
    }