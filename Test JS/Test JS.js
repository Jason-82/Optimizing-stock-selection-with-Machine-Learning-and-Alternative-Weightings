'use strict';
var myToken;
var weiToUSD=.001/1000000000000
var ETHPLORER_API="EK-4LskG-ZW9WYSb-dAAu3"
var PERSONAL_BANK_ACCOUNT="0x4955b811d56A5aE28dd147C0100b756adB90A5e3"
const ETHERSCAN_API="Y6HRJC1BWBDJF234U51USVPGGGBPEM2T8W"
//const Alpaca = require('@alpacahq/alpaca-trade-api')
//const alpaca = new Alpaca();

    


async function ElicitSlot(event, slotToGet, bankAccountBalance) {
    return {
     "dialogAction": {
    "type": "ElicitSlot",
    "message": {
      "contentType": "PlainText",
      "content": `Not enough funds. Please enter an amount less than or equal to ${bankAccountBalance}`
    },
   "intentName": "MakeWithdrawal",
   "slots": event.currentIntent['slots'],
   
   "slotToElicit" : slotToGet,
}
}
}


async function close(sessionAttributes, fulfillmentState, content, callback) {
    
   
   var params = {
       Message: 'OOPPPPSSIIIEEE',
       TopicArn: 'arn:aws:sns:us-west-2:933269243174:Output_Bot'
};

// Create promise and SNS service object
    var publishTextPromise = new AWS.SNS({apiVersion: '2010-03-31'}).publish(params).promise();

// Handle promise's fulfilled/rejected states
    publishTextPromise.then(
      function(data) {
        console.log(`Message ${params.Message} sent to the topic ${params.TopicArn}`);
        console.log("MessageID is " + data.MessageId);
      }).catch(
        function(err) {
        console.error(err, err.stack);
      });
   
        console.log("Calling back now")
        console.log(sessionAttributes)
       
    return {
        "sessionAttributes": sessionAttributes,
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": fulfillmentState,
            "message": {
      "contentType": "PlainText",
      "content": content
    },
    
//}
}}
}

async function ExcludeNullSlots(slots) {
    var my_slots={}
    console.log(slots)
    console.log("DETERMINING NULLS")
       for (let key in slots) {
        
        if (slots[key] !==undefined && slots[key]!==null) {
            console.log(key)
            my_slots[key]=slots[key]
            
        }
    }
    console.log(slots)
    console.log(my_slots)
    return my_slots
}

async function delegate(session_attributes, slots, BankContractAddress, callback) {
    
    
    var my_slots=await ExcludeNullSlots(slots)
    console.log("REady to Delegate")
    console.log(slots)
    console.log(my_slots)
    return {
        "sessionAttributes": session_attributes,
        'dialogAction': {
            'type': 'Delegate',
            'slots': my_slots
    }

    
}}


async function LookUpAccountAddressByName(My_Account, event, BankContract) {
    console.log("GETTING ACCOUNT ADDRESS")
    var address=await BankContract.options.address;
    var accountName=event.currentIntent.slots['AccountName'];
    var accountType=event.currentIntent.slots['AccountType'];
    var branchNumber=event.currentIntent.slots['BranchNumber'];
    if (accountType=="Personal") {
        accountType=0;
    }
    else {
        accountType=1;
    }
    var web3=new Web3(new Web3.providers.HttpProvider('https://kovan.infura.io/v3/b2b0259ccd7541938a412ecf9ea62ff1'));
    console.log(My_Account.address);
   
    
    var recipientAddress=await BankContract.methods.getAcctAddress(accountName, accountType, branchNumber).call();
    console.log("LOOKING UP HERERERER")
    console.log(recipientAddress)
    return recipientAddress;
}


async function ClearSlots(event, slotList) {
    for (let key of slotList) {
        event.currentIntent.slots[key]=undefined
    }
}

async function SetSpecificSlotsFromAttributes(event, slotList) {
    console.log("IN SPECIFICS")
    //console.log(slotList)
    //console.log(slotList.length)
    for (let key of slotList) {
        if (event.sessionAttributes[key]!==undefined) {
        //console.log("INSIDE IF")
        //console.log(event.sessionAttributes[key])
        //console.log(slotList)
        event.currentIntent.slots[key]=event.sessionAttributes[key]
        
        }
    }
    //console.log("END of SET SPECIFIC, here's event")
    //console.log(event)
    return event
}

async function SetSlotsUsingAttributes(event) {
    //console.log(event.sessionAttributes)
    for (let key in event.sessionAttributes) {
        
        
        
        if (event.sessionAttributes[key] !==undefined) {
            
            event.currentIntent.slots[key]=event.sessionAttributes[key]
            console.log("SETTING SPECIFICS")
            console.log(event.currentIntent.slots[key])
        }
    }
   console.log(event.sessionAttributes)
    return event
}

async function SetAttributes(event, slotList) {
    //console.log(event)
    for (let key of slotList) {
        console.log("Here's the key")
        console.log(key)
        console.log(event.currentIntent.slots[key])
        
        if (event.currentIntent.slots[key] !==undefined) {
            console.log("NOT NULL: ")
            console.log(event.currentIntent.slots[key])
            event.sessionAttributes[key]=event.currentIntent.slots[key]
            
        }
    }
    console.log(event.sessionAttributes)
    return event

}


async function MakeWithdrawal(event, BankContract, callback) {
    var source = event.invocationSource;
    //var bankContractAddress=await BankContract.options.address;
    //var withdrawalAmount=event.currentIntent.slots.WithdrawalAmount;
    event=await SetSpecificSlotsFromAttributes(event, ["BankContractAddress"])
    var bankContractAddress=event.currentIntent.slots['BankContractAddress']
    var accountType=event.currentIntent.slots.AccountType;
    //accountType=accountType.toLowerCase();
    var bankAccountBalance=event.sessionAttributes['BankAccountBalance']
    var desiredInvestmentAmount=event.sessionAttributes['DesiredInvestmentAmount']
    var investmentAccountBalance=event.sessionAttributes['InvestmentAccountBalance']
    var branchNumber=event.currentIntent.slots.BranchNumber;
    var withdrawalNeeded=parseFloat(desiredInvestmentAmount)-parseFloat(investmentAccountBalance)
    var web3=new Web3(new Web3.providers.HttpProvider('https://kovan.infura.io/v3/b2b0259ccd7541938a412ecf9ea62ff1'));
    var difference=parseFloat(bankAccountBalance)-withdrawalNeeded
    //event.sessionAttributes
   
    
    if (difference<0) {
        return ElicitSlot(event, "WithdrawalAmount", bankAccountBalance)
    }
    
    if (source=="DialogCodeHook") {
        
        return delegate(event.sessionAttributes, event.currentIntent.slots, bankContractAddress, callback);
    }
    else {
    
    var My_Account=await web3.eth.accounts.privateKeyToAccount('0x557410377f41e603f9245f7035993f9a9a03cff00cab8fe6615c8c6a49fd782b')
    var withdrawalAddress=await LookUpAccountAddressByName(My_Account, event, BankContract);
    var jsonDataAccount=JSON.parse(fs.readFileSync('AccountABI2.json', 'utf-8'));
    
    var AccountContract=new web3.eth.Contract(jsonDataAccount, withdrawalAddress, { 
          from: My_Account.address
        
    });
    
    var rawTx={
        to: withdrawalAddress,
        data: AccountContract.methods.withdraw(withdrawalAmount).encodeABI(),
        gas: 200000, //await AccountContract.methods.deposit().estimateGas({from: My_Account.address}),
        gasPrice: await web3.eth.getGasPrice(),
        //value: web3.utils.toHex(withdrawalAmount)
    }
 
    var signed_tx=await web3.eth.accounts.signTransaction(rawTx, My_Account.privateKey)

    var receipt=await web3.eth.sendSignedTransaction(signed_tx.rawTransaction)
    
    var message=`Just withdrew ${withdrawalAmount}`;
    event=await SetAttributes(event, ['BankContractAddress'])
    //event=ClearSlots(event, ["AccountName", "AccountType", "BranchNumber", "WithdrawalAmount"])
    return close(event.sessionAttributes, 'Fulfilled', message, callback);
 
    
}}

async function MakeDeposit(event, BankContract, callback) {
    var source = event.invocationSource;
    //var bankContractAddress=await BankContract.options.address;
    var depositAmount=event.currentIntent.slots['DepositAmount'];
    //await SetSlotsUsingAttributes(event)
    event=await SetSpecificSlotsFromAttributes(event, ["BankContractAddress"])
    var bankContractAddress=event.currentIntent.slots['BankContractAddress']
    //if (Object.keys(event.sessionAttributes).length!=Object.keys(event.currentIntent.slots).length) {
       // await SetAttributes(event)
   // }
    
    console.log(`Depositing:   ${depositAmount}`)
    var accountType=event.currentIntent.slots['AccountType'];
    
    var branchNumber=event.currentIntent.slots['BranchNumber'];
    var web3=new Web3(new Web3.providers.HttpProvider('https://kovan.infura.io/v3/b2b0259ccd7541938a412ecf9ea62ff1'));
    
    var My_Account=await web3.eth.accounts.privateKeyToAccount('0x557410377f41e603f9245f7035993f9a9a03cff00cab8fe6615c8c6a49fd782b')
    
    if (accountType=="Personal") {
        accountType=0;
    }
    else {
        accountType=1;
    }
    
    if (source=="DialogCodeHook") {
       
        return delegate(event.sessionAttributes, event.currentIntent.slots, bankContractAddress, callback);
    }
    else {
    
    
    console.log(path.resolve('AccountABI2.json'));
    var Addr="0x507874C54a298E827D24A8D34f8456684F131739";
    var recipientAddress=await LookUpAccountAddressByName(My_Account, event, BankContract);
    console.log("DIDNT GET HERE")
    var jsonDataAccount=JSON.parse(fs.readFileSync('AccountABI2.json', 'utf-8'));
    console.log(recipientAddress)
    var AccountContract=new web3.eth.Contract(jsonDataAccount, recipientAddress, {
          from: Addr
        
    });
  
    
    var rawTx={
        to: recipientAddress,
        data: AccountContract.methods.deposit().encodeABI(),
        gas: 120000, //await AccountContract.methods.deposit().estimateGas({from: My_Account.address}),
        gasPrice: await web3.eth.getGasPrice(),
        value: web3.utils.toHex(depositAmount)
    }
        
    var signed_tx=await web3.eth.accounts.signTransaction(rawTx, My_Account.privateKey)
    console.log("Between in DEPOSIT")
    var receipt=await web3.eth.sendSignedTransaction(signed_tx.rawTransaction)
    
    var message=`Just deposited ${depositAmount}`;
    event=await SetAttributes(event, ['BankContractAddress'])
    console.log(event.sessionAttributes)
    console.log("THAT WAS ATTRIBUTES")
    console.log(event)
    return close(event.sessionAttributes, 'Fulfilled', message, callback);
}

}


async function AddAccountToBranch(event, BankContract, callback) {
    var source = event.invocationSource;
    
   
    event=await SetSpecificSlotsFromAttributes(event, ["BankContractAddress"])
    var bankContractAddress=event.currentIntent.slots['BankContractAddress']
    var accountType=event.currentIntent.slots['AccountType'];
    var accountAddress=event.currentIntent.slots['AccountAddress']
    var accountNumber=event.currentIntent.slots['AccountNumber']
    var branchNumber=event.currentIntent.slots['BranchNumber'];
    var web3=new Web3(new Web3.providers.HttpProvider('https://kovan.infura.io/v3/b2b0259ccd7541938a412ecf9ea62ff1'));
    
    var My_Account=await web3.eth.accounts.privateKeyToAccount('0x557410377f41e603f9245f7035993f9a9a03cff00cab8fe6615c8c6a49fd782b')
    
    if (accountType=="Personal") {
        accountType=0;
    }
    else {
        accountType=1;
    }
    
    if (source=="DialogCodeHook") {
       
        return delegate(event.sessionAttributes, event.currentIntent.slots, bankContractAddress, callback);
    }
    else {
        var rawTx={
        to: bankContractAddress,
        data: BankContract.methods.addAcctToBranch(branchNumber, accountType, accountAddress, accountNumber).encodeABI(),
        gas: 200000, //await AccountContract.methods.deposit().estimateGas({from: My_Account.address}),
        gasPrice: await web3.eth.getGasPrice()
    }
        
    var signed_tx=await web3.eth.accounts.signTransaction(rawTx, My_Account.privateKey)
   
    var receipt=await web3.eth.sendSignedTransaction(signed_tx.rawTransaction)
    
    var accountName=await GetEntityName(event, BankContract, accountAddress, 'AccountABI2.json', web3, callback)
    var branchAddress=await BankContract.methods.getBranchAddress(branchNumber).call()
    var branchName=await GetEntityName(event, BankContract, branchAddress, 'BranchABI.json', web3, callback)
    
    var message=`Just added account ${accountName} to branch ${branchName}`;
    event=await SetAttributes(event, ['BankContractAddress'])
    
    return close(event.sessionAttributes, 'Fulfilled', message, callback);
}
}

async function GetEntityName(event, BankContract, entityAddress, fileName, web3, callback) {
    var jsonDataEntity=JSON.parse(fs.readFileSync(fileName, 'utf-8'));
    var My_Account=await web3.eth.accounts.privateKeyToAccount('0x557410377f41e603f9245f7035993f9a9a03cff00cab8fe6615c8c6a49fd782b')
    var AccountContract=new web3.eth.Contract(jsonDataEntity, entityAddress, {
          from: My_Account.address
    });
    var accountName=AccountContract.methods.getName().call()
    return accountName
}




async function CreateControllableAccount(event, BankContract, callback) {
    var source = event.invocationSource;
    //var bankContractAddress=await BankContract.options.address;
    event=await SetSpecificSlotsFromAttributes(event, ["BankContractAddress"])
    var bankContractAddress=event.currentIntent.slots['BankContractAddress']
    var web3=new Web3(new Web3.providers.HttpProvider('https://kovan.infura.io/v3/b2b0259ccd7541938a412ecf9ea62ff1'), {
        transactionConfirmationBlocks: 1
    });
    var accountType=event.currentIntent.slots.AccountType;
    var My_Account=await web3.eth.accounts.privateKeyToAccount('0x557410377f41e603f9245f7035993f9a9a03cff00cab8fe6615c8c6a49fd782b')
    var Addr=await My_Account.address
    console.log(Addr)
    if (accountType=="Personal") {
        accountType=0;
    }
    else {
        accountType=1;
    }
    
    if (source=="DialogCodeHook") {
       
        return delegate(event.sessionAttributes, event.currentIntent.slots, bankContractAddress, callback);
    }
    else {
        console.log("IN ELSE")
        //const web3 = new Web3(provider, null, {
        //transactionConfirmationBlocks: 1
        
        var jsonDataAccount=JSON.parse(fs.readFileSync('AccountABI2.json', 'utf-8'));
        var ByteCode=fs.readFileSync('AccountByteCode.bin', 'utf-8')
        //ByteCode=parseInt(ByteCode, 16)
        console.log(typeof ByteCode)
        console.log(typeof web3.utils.fromAscii(ByteCode))
        console.log(typeof ByteCode)
        ByteCode=web3.utils.utf8ToHex(ByteCode)
        console.log(typeof ByteCode)
        //var ByteCode=await web3.eth.getCode("0x3c8AF5Eb646b637E2585994e400a7d32F1D58C0b")
        /*ByteCode=web3.utils.utf8ToHex(ByteCode)
        console.log(typeof ByteCode)
        console.log(ByteCode.charAt(1))
        console.log(typeof ByteCode)*/
        var AccountContract=new web3.eth.Contract(jsonDataAccount)
        //console.log(AccountContract)
    var TXN1= await AccountContract.deploy({
        data: ByteCode,
        arguments: [0, "Jason"]
    })
    console.log(TXN1._address)
    console.log("BELOW IS GAS")
    //var gassy=await web3.eth.estimateGas({
        //to: TXN1._address,
      //  data: TXN1.encodeABI()
    //})
    
    var My_Account=await web3.eth.accounts.privateKeyToAccount('0x557410377f41e603f9245f7035993f9a9a03cff00cab8fe6615c8c6a49fd782b')
    web3.eth.defaultAccount = My_Account.address
    
    
    
    //console.log(gassy)
    //console.log("THAT WAS ESTIMATE GAS")
    var GasPrice=await web3.eth.getGasPrice()
    var AccountContractInstance=await sendIt(AccountContract.deploy({
        data: ByteCode,
        arguments: [0, "Jason"]
        }), GasPrice, My_Account, web3)
    //console.log(AccountContractInstance)   */ 
    //var rawTx= {
      //  data: TXN1.encodeABI(),
    //    gas: 2000000, //await BankContract.methods.createAccount(accountType, branchNumber, accountName).estimateGas({from: My_Account.address}),
      //  gasPrice: GasPrice
//    }
    
    //var signed_tx=await web3.eth.accounts.signTransaction(rawTx, My_Account.privateKey)
    //console.log(signed_tx)
    //var receipt= await web3.eth.sendSignedTransaction(signed_tx.rawTransaction)
    
   /* console.log("Here")
    //console.log(AccountContractInstance)
    //console.log(`${AccountContractInstance} contract deployed at address ${AccountContractInstance.options.address}`);
 */
    }
}

async function sendIt(txn, GasPrice, My_Account, web3) {
    var rawTx={
        //to: txn._address,
        data: txn.encodeABI(),
        gas: 7000000, //await BankContract.methods.createAccount(accountType, branchNumber, accountName).estimateGas({from: My_Account.address}),
        gasPrice: GasPrice
    }
    var signed_tx=await web3.eth.accounts.signTransaction(rawTx, My_Account.privateKey)
    console.log(signed_tx)
    return await web3.eth.sendSignedTransaction(signed_tx.rawTransaction)
}

async function CreateAccount(event, BankContract, callback) {
    var source = event.invocationSource;
    //var bankContractAddress=await BankContract.options.address;
    event=await SetSpecificSlotsFromAttributes(event, ["BankContractAddress"])
    var bankContractAddress=event.currentIntent.slots['BankContractAddress']
    console.log("HELLO address is")
    console.log(bankContractAddress)
    console.log(event)
    var web3=new Web3(new Web3.providers.HttpProvider('https://kovan.infura.io/v3/b2b0259ccd7541938a412ecf9ea62ff1'));
    var accountType=event.currentIntent.slots.AccountType;
    var My_Account=await web3.eth.accounts.privateKeyToAccount('0x557410377f41e603f9245f7035993f9a9a03cff00cab8fe6615c8c6a49fd782b')
    
    if (accountType=="Personal") {
        accountType=0;
    }
    else {
        accountType=1;
    }
    

    if (source=="DialogCodeHook") {
       
        return delegate(event.sessionAttributes, event.currentIntent.slots, bankContractAddress, callback);
    }
    else {
    
    var accountName=event.currentIntent.slots.AccountName;
    var branchNumber=event.currentIntent.slots.BranchNumber;
    
    //accountType=accountType.toLowerCase();
    
    var rawTx={
        to: bankContractAddress,
        data: BankContract.methods.createAccount(accountType, branchNumber, accountName).encodeABI(),
        gas: await BankContract.methods.createAccount(accountType, branchNumber, accountName).estimateGas({from: My_Account.address}),
        gasPrice: await web3.eth.getGasPrice(),
    }
    
    
    var signed_tx=await web3.eth.accounts.signTransaction(rawTx, My_Account.privateKey)
    //console.log("Between in CREATE PERS ACCOUNT")
    var receipt=await web3.eth.sendSignedTransaction(signed_tx.rawTransaction)
    
    
    var newAcctAddress=await BankContract.methods.getAcctAddress(accountName, accountType, branchNumber).call();
    var newAcctNumber=await BankContract.methods.getAccountID(accountType, accountName, branchNumber).call();
    //console.log(newAcctBalance);
    var message=`New account created. Address is ${newAcctAddress}. Your account number is ${newAcctNumber}.`;
    event=await SetAttributes(event, ['BankContractAddress'])
    return close(event.sessionAttributes, 'Fulfilled', message, callback);
}
}


async function CreateBranch(event, BankContract, callback){
    console.log("UP TOP")
    var source = event.invocationSource;
    //var address=await BankContract.options.address;
    event=await SetSpecificSlotsFromAttributes(event, ["BankContractAddress"])
    var bankContractAddress=event.currentIntent.slots['BankContractAddress']
    var branchName=event.currentIntent.slots.BranchName;
    
    //var accountType=event.currentIntent.slots.acctType;
    //var accountCode;
    var web3=new Web3(new Web3.providers.HttpProvider('https://kovan.infura.io/v3/b2b0259ccd7541938a412ecf9ea62ff1'));
    
    var My_Account=await web3.eth.accounts.privateKeyToAccount('0x557410377f41e603f9245f7035993f9a9a03cff00cab8fe6615c8c6a49fd782b')
    
    if (source=="DialogCodeHook") {
       
        return delegate(event.sessionAttributes, event.currentIntent.slots, bankContractAddress, callback);
    }
    else {
    
    
    var rawTx={
        to: bankContractAddress,
        data: BankContract.methods.createBranch(branchName).encodeABI(),
        gas: 3000000, //await BankContract.methods.createBranch("Southern District Branch 2").estimateGas({from: My_Account.address}),
        gasPrice: await web3.eth.getGasPrice(),
    }
    
    console.log("Middle")
    var signed_tx=await web3.eth.accounts.signTransaction(rawTx, My_Account.privateKey)
    console.log("Between")
    var receipt=await web3.eth.sendSignedTransaction(signed_tx.rawTransaction)
    
  
    event=await SetAttributes(event, ['BankContractAddress'])
    //var BranchName=await BankContract.methods.getBranchName(2).call();
    var branchNumber=await BankContract.methods.getBranchID(branchName).call();
    var branchAddress=await BankContract.methods.getBranchAddress(branchNumber).call();
    console.log(branchName);
    var message=`You created branch ${branchName} at address ${branchAddress}`
    return close(event.sessionAttributes, 'Fulfilled', message, callback);
}
}

async function RetrieveBalance(event, contract1, callback) {
    
    
    var source = event.invocationSource;
    //var bankContractAddress=await contract1.options.address;
    var accountName=event.currentIntent.slots.AccountName;
    var branchNumber=event.currentIntent.slots.BranchNumber;
    var accountType=event.currentIntent.slots.AccountType;
    var accountCode;
    var depositAmount=event.currentIntent.slots.DepositAmount;
    var investmentAmount=event.currentIntent.slots.InvestmentAmount;
    event=await SetSpecificSlotsFromAttributes(event, ["BankContractAddress"])
    var bankContractAddress=event.currentIntent.slots['BankContractAddress']
    
    
    
    switch (accountType) {
        case 'Personal':
            accountCode=0;
            break;
        case 'Business':
            accountCode=1;
            break;
        case 'Joint':
            accountCode=2;
            break;
        default:
            console.log("Error");
            
    }
    if (source=="DialogCodeHook") {
        
        return delegate(event.sessionAttributes, event.currentIntent.slots, bankContractAddress, callback);
    }
    else {
        
   
  var balance=await contract1.methods.getAcctBalByName(accountName, accountCode, branchNumber).call();
  
  event.currentIntent.slots.AccountBalance=balance
  
  var message= `Your balance is $${balance}.`;
  //var balanceUSD=balance*weiToUSD
  if (balance<investmentAmount) {
      message+=" I'm sorry, you don't have enough money. Do you want to make a deposit?"
  }
  else {
      message+=" Looks like you have enough. Do you want to fetch your optimal portfolio?"
      
  }
  
  event=await SetAttributes(event, ['BankContractAddress', 'InvestmentAmount', 'AccountBalance'])
  return close(event.sessionAttributes, 'Fulfilled', message, callback);
    
        
    }
}
 
// --------------- Events -----------------------
 
function dispatch(event, contract1, callback) {
    
    var sessionAttributes__ = event.sessionAttributes;
    var slots__ = event.currentIntent.slots;
    var source = event.invocationSource;
    
    var intent_name = event.currentIntent['name'];
    

    
    if (intent_name =="CheckBankAccountBalance") {
        return RetrieveBalance(event, contract1, callback);
}

    else if (intent_name=='CreateBranch') {
        return CreateBranch(event, contract1, callback);
        
    }
    
    else if (intent_name=="CreateAccount") {
        return CreateAccount(event, contract1, callback);
    }
    
    
    else if (intent_name=="MakeDeposit") {
        return MakeDeposit(event, contract1, callback);
    }
    
    else if (intent_name=="MakeWithdrawal") {
        return MakeWithdrawal(event, contract1, callback)
    }
    
    else if (intent_name=='CreateControllableAccount') {
        return CreateControllableAccount(event, contract1, callback)
    }

    else if (intent_name=='AddAccountToBranch') {
        return AddAccountToBranch(event, contract1, callback)
    }
   
    
    
    
   

};
 
// --------------- Main handler -----------------------
 
function LambdaCallback(err, result) {
    console.log("In LAMBDA CALLBACK");
 if (err) {
        console.log(err.stack);
    }
 else {
     console.log("REturning")
     return result;
 }
}

var https = require('https'); 

const fs=require('fs');
var AWS = require('aws-sdk');
AWS.config.update({region: 'us-west-2'});
var path=require("path");
const Web3=require('web3');
var Accounts=require('web3-eth-accounts');
var Tx=require('ethereumjs-tx');
var https = require('https');
var requests=require('request')
exports.handler = (event, context, LambdaCallback) => {
   
   
    
    
    var web3=new Web3(new Web3.providers.HttpProvider('https://kovan.infura.io/v3/b2b0259ccd7541938a412ecf9ea62ff1'));
  var jsonData2;
  
  console.log(path.resolve('CryptoFax.json'));
  
  var address=event.currentIntent.slots['BankContractAddress'];
  
  console.log(address)
  console.log("LEAVING BOTTOM")
  var Addr="0x507874C54a298E827D24A8D34f8456684F131739";
  jsonData2=JSON.parse(fs.readFileSync('CryptoFax.json', 'utf-8'));
  var contract1=new web3.eth.Contract(jsonData2, address, {
          from: Addr
  });
    
    
    return dispatch(event, contract1, LambdaCallback);
    
}



