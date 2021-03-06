from fbprophet import Prophet
import alpaca_trade_api as tradeapi
import yfinance as yf
import numpy as np
import json
import pandas as pd
#import Risk_Kit2 as rk
import boto3
import datetime as dt
import os
pd.plotting.register_matplotlib_converters()  # https://github.com/facebook/prophet/issues/999
s3 = boto3.client('s3')
s4 = boto3.resource(service_name='s3', region_name='us-west-2')

#my_data=pd.DataFrame()

def get_constituents(alreadyInvestedStocks):
  #sp500 = [{"Name": "3M Company", "Sector": "Industrials", "Symbol": "MMM"},{"Name": "A.O. Smith Corp", "Sector": "Industrials", "Symbol": "AOS"},{"Name": "Abbott Laboratories", "Sector": "Health Care", "Symbol": "ABT"},{"Name": "AbbVie Inc.", "Sector": "Health Care", "Symbol": "ABBV"},{"Name": "Accenture plc", "Sector": "Information Technology", "Symbol": "ACN"},{"Name": "Activision Blizzard", "Sector": "Information Technology", "Symbol": "ATVI"},{"Name": "Acuity Brands Inc", "Sector": "Industrials", "Symbol": "AYI"},{"Name": "Adobe Systems Inc", "Sector": "Information Technology", "Symbol": "ADBE"},{"Name": "Advance Auto Parts", "Sector": "Consumer Discretionary", "Symbol": "AAP"},{"Name": "Advanced Micro Devices Inc", "Sector": "Information Technology", "Symbol": "AMD"},{"Name": "AES Corp", "Sector": "Utilities", "Symbol": "AES"},{"Name": "Aetna Inc", "Sector": "Health Care", "Symbol": "AET"},{"Name": "Affiliated Managers Group Inc", "Sector": "Financials", "Symbol": "AMG"},{"Name": "AFLAC Inc", "Sector": "Financials", "Symbol": "AFL"},{"Name": "Agilent Technologies Inc", "Sector": "Health Care", "Symbol": "A"},{"Name": "Air Products & Chemicals Inc", "Sector": "Materials", "Symbol": "APD"},{"Name": "Akamai Technologies Inc", "Sector": "Information Technology", "Symbol": "AKAM"},{"Name": "Alaska Air Group Inc", "Sector": "Industrials", "Symbol": "ALK"},{"Name": "Albemarle Corp", "Sector": "Materials", "Symbol": "ALB"},{"Name": "Alexandria Real Estate Equities Inc", "Sector": "Real Estate", "Symbol": "ARE"},{"Name": "Alexion Pharmaceuticals", "Sector": "Health Care", "Symbol": "ALXN"},{"Name": "Align Technology", "Sector": "Health Care", "Symbol": "ALGN"},{"Name": "Allegion", "Sector": "Industrials", "Symbol": "ALLE"},{"Name": "Allergan, Plc", "Sector": "Health Care", "Symbol": "AGN"},{"Name": "Alliance Data Systems", "Sector": "Information Technology", "Symbol": "ADS"},{"Name": "Alliant Energy Corp", "Sector": "Utilities", "Symbol": "LNT"},{"Name": "Allstate Corp", "Sector": "Financials", "Symbol": "ALL"},{"Name": "Alphabet Inc Class A", "Sector": "Information Technology", "Symbol": "GOOGL"},{"Name": "Alphabet Inc Class C", "Sector": "Information Technology", "Symbol": "GOOG"},{"Name": "Altria Group Inc", "Sector": "Consumer Staples", "Symbol": "MO"},{"Name": "Amazon.com Inc.", "Sector": "Consumer Discretionary", "Symbol": "AMZN"},{"Name": "Ameren Corp", "Sector": "Utilities", "Symbol": "AEE"},{"Name": "American Airlines Group", "Sector": "Industrials", "Symbol": "AAL"},{"Name": "American Electric Power", "Sector": "Utilities", "Symbol": "AEP"},{"Name": "American Express Co", "Sector": "Financials", "Symbol": "AXP"},{"Name": "American International Group, Inc.", "Sector": "Financials", "Symbol": "AIG"},{"Name": "American Tower Corp A", "Sector": "Real Estate", "Symbol": "AMT"},{"Name": "American Water Works Company Inc", "Sector": "Utilities", "Symbol": "AWK"},{"Name": "Ameriprise Financial", "Sector": "Financials", "Symbol": "AMP"},{"Name": "AmerisourceBergen Corp", "Sector": "Health Care", "Symbol": "ABC"},{"Name": "AMETEK Inc.", "Sector": "Industrials", "Symbol": "AME"},{"Name": "Amgen Inc.", "Sector": "Health Care", "Symbol": "AMGN"},{"Name": "Amphenol Corp", "Sector": "Information Technology", "Symbol": "APH"},{"Name": "Anadarko Petroleum Corp", "Sector": "Energy", "Symbol": "APC"},{"Name": "Analog Devices, Inc.", "Sector": "Information Technology", "Symbol": "ADI"},{"Name": "Andeavor", "Sector": "Energy", "Symbol": "ANDV"},{"Name": "ANSYS", "Sector": "Information Technology", "Symbol": "ANSS"},{"Name": "Anthem Inc.", "Sector": "Health Care", "Symbol": "ANTM"},{"Name": "Aon plc", "Sector": "Financials", "Symbol": "AON"},{"Name": "Apache Corporation", "Sector": "Energy", "Symbol": "APA"},{"Name": "Apartment Investment & Management", "Sector": "Real Estate", "Symbol": "AIV"},{"Name": "Apple Inc.", "Sector": "Information Technology", "Symbol": "AAPL"},{"Name": "Applied Materials Inc.", "Sector": "Information Technology", "Symbol": "AMAT"},{"Name": "Aptiv Plc", "Sector": "Consumer Discretionary", "Symbol": "APTV"},{"Name": "Archer-Daniels-Midland Co", "Sector": "Consumer Staples", "Symbol": "ADM"},{"Name": "Arconic Inc.", "Sector": "Industrials", "Symbol": "ARNC"},{"Name": "Arthur J. Gallagher & Co.", "Sector": "Financials", "Symbol": "AJG"},{"Name": "Assurant Inc.", "Sector": "Financials", "Symbol": "AIZ"},{"Name": "AT&T Inc.", "Sector": "Telecommunication Services", "Symbol": "T"},{"Name": "Autodesk Inc.", "Sector": "Information Technology", "Symbol": "ADSK"},{"Name": "Automatic Data Processing", "Sector": "Information Technology", "Symbol": "ADP"},{"Name": "AutoZone Inc", "Sector": "Consumer Discretionary", "Symbol": "AZO"},{"Name": "AvalonBay Communities, Inc.", "Sector": "Real Estate", "Symbol": "AVB"},{"Name": "Avery Dennison Corp", "Sector": "Materials", "Symbol": "AVY"},{"Name": "Baker Hughes, a GE Company", "Sector": "Energy", "Symbol": "BHGE"},{"Name": "Ball Corp", "Sector": "Materials", "Symbol": "BLL"},{"Name": "Bank of America Corp", "Sector": "Financials", "Symbol": "BAC"},{"Name": "Baxter International Inc.", "Sector": "Health Care", "Symbol": "BAX"},{"Name": "BB&T Corporation", "Sector": "Financials", "Symbol": "BBT"},{"Name": "Becton Dickinson", "Sector": "Health Care", "Symbol": "BDX"},{"Name": "Berkshire Hathaway", "Sector": "Financials", "Symbol": "BRK.B"},{"Name": "Best Buy Co. Inc.", "Sector": "Consumer Discretionary", "Symbol": "BBY"},{"Name": "Biogen Inc.", "Sector": "Health Care", "Symbol": "BIIB"},{"Name": "BlackRock", "Sector": "Financials", "Symbol": "BLK"},{"Name": "Block H&R", "Sector": "Financials", "Symbol": "HRB"},{"Name": "Boeing Company", "Sector": "Industrials", "Symbol": "BA"},{"Name": "Booking Holdings Inc", "Sector": "Consumer Discretionary", "Symbol": "BKNG"},{"Name": "BorgWarner", "Sector": "Consumer Discretionary", "Symbol": "BWA"},{"Name": "Boston Properties", "Sector": "Real Estate", "Symbol": "BXP"},{"Name": "Boston Scientific", "Sector": "Health Care", "Symbol": "BSX"},{"Name": "Brighthouse Financial Inc", "Sector": "Financials", "Symbol": "BHF"},{"Name": "Bristol-Myers Squibb", "Sector": "Health Care", "Symbol": "BMY"},{"Name": "Broadcom", "Sector": "Information Technology", "Symbol": "AVGO"},{"Name": "Brown-Forman Corp.", "Sector": "Consumer Staples", "Symbol": "BF.B"},{"Name": "C. H. Robinson Worldwide", "Sector": "Industrials", "Symbol": "CHRW"},{"Name": "CA, Inc.", "Sector": "Information Technology", "Symbol": "CA"},{"Name": "Cabot Oil & Gas", "Sector": "Energy", "Symbol": "COG"},{"Name": "Cadence Design Systems", "Sector": "Information Technology", "Symbol": "CDNS"},{"Name": "Campbell Soup", "Sector": "Consumer Staples", "Symbol": "CPB"},{"Name": "Capital One Financial", "Sector": "Financials", "Symbol": "COF"},{"Name": "Cardinal Health Inc.", "Sector": "Health Care", "Symbol": "CAH"},{"Name": "Carmax Inc", "Sector": "Consumer Discretionary", "Symbol": "KMX"},{"Name": "Carnival Corp.", "Sector": "Consumer Discretionary", "Symbol": "CCL"},{"Name": "Caterpillar Inc.", "Sector": "Industrials", "Symbol": "CAT"},{"Name": "Cboe Global Markets", "Sector": "Financials", "Symbol": "CBOE"},{"Name": "CBRE Group", "Sector": "Real Estate", "Symbol": "CBRE"},{"Name": "CBS Corp.", "Sector": "Consumer Discretionary", "Symbol": "CBS"},{"Name": "Celgene Corp.", "Sector": "Health Care", "Symbol": "CELG"},{"Name": "Centene Corporation", "Sector": "Health Care", "Symbol": "CNC"},{"Name": "CenterPoint Energy", "Sector": "Utilities", "Symbol": "CNP"},{"Name": "CenturyLink Inc", "Sector": "Telecommunication Services", "Symbol": "CTL"},{"Name": "Cerner", "Sector": "Health Care", "Symbol": "CERN"},{"Name": "CF Industries Holdings Inc", "Sector": "Materials", "Symbol": "CF"},{"Name": "Charles Schwab Corporation", "Sector": "Financials", "Symbol": "SCHW"},{"Name": "Charter Communications", "Sector": "Consumer Discretionary", "Symbol": "CHTR"},{"Name": "Chevron Corp.", "Sector": "Energy", "Symbol": "CVX"},{"Name": "Chipotle Mexican Grill", "Sector": "Consumer Discretionary", "Symbol": "CMG"},{"Name": "Chubb Limited", "Sector": "Financials", "Symbol": "CB"},{"Name": "Church & Dwight", "Sector": "Consumer Staples", "Symbol": "CHD"},{"Name": "CIGNA Corp.", "Sector": "Health Care", "Symbol": "CI"},{"Name": "Cimarex Energy", "Sector": "Energy", "Symbol": "XEC"},{"Name": "Cincinnati Financial", "Sector": "Financials", "Symbol": "CINF"},{"Name": "Cintas Corporation", "Sector": "Industrials", "Symbol": "CTAS"},{"Name": "Cisco Systems", "Sector": "Information Technology", "Symbol": "CSCO"},{"Name": "Citigroup Inc.", "Sector": "Financials", "Symbol": "C"},{"Name": "Citizens Financial Group", "Sector": "Financials", "Symbol": "CFG"},{"Name": "Citrix Systems", "Sector": "Information Technology", "Symbol": "CTXS"},{"Name": "CME Group Inc.", "Sector": "Financials", "Symbol": "CME"},{"Name": "CMS Energy", "Sector": "Utilities", "Symbol": "CMS"},{"Name": "Coca-Cola Company (The)", "Sector": "Consumer Staples", "Symbol": "KO"},{"Name": "Cognizant Technology Solutions", "Sector": "Information Technology", "Symbol": "CTSH"},{"Name": "Colgate-Palmolive", "Sector": "Consumer Staples", "Symbol": "CL"},{"Name": "Comcast Corp.", "Sector": "Consumer Discretionary", "Symbol": "CMCSA"},{"Name": "Comerica Inc.", "Sector": "Financials", "Symbol": "CMA"},{"Name": "Conagra Brands", "Sector": "Consumer Staples", "Symbol": "CAG"},{"Name": "Concho Resources", "Sector": "Energy", "Symbol": "CXO"},{"Name": "ConocoPhillips", "Sector": "Energy", "Symbol": "COP"},{"Name": "Consolidated Edison", "Sector": "Utilities", "Symbol": "ED"},{"Name": "Constellation Brands", "Sector": "Consumer Staples", "Symbol": "STZ"},{"Name": "Corning Inc.", "Sector": "Information Technology", "Symbol": "GLW"},{"Name": "Costco Wholesale Corp.", "Sector": "Consumer Staples", "Symbol": "COST"},{"Name": "Coty, Inc", "Sector": "Consumer Staples", "Symbol": "COTY"},{"Name": "Crown Castle International Corp.", "Sector": "Real Estate", "Symbol": "CCI"},{"Name": "CSRA Inc.", "Sector": "Information Technology", "Symbol": "CSRA"},{"Name": "CSX Corp.", "Sector": "Industrials", "Symbol": "CSX"},{"Name": "Cummins Inc.", "Sector": "Industrials", "Symbol": "CMI"},{"Name": "CVS Health", "Sector": "Consumer Staples", "Symbol": "CVS"},{"Name": "D. R. Horton", "Sector": "Consumer Discretionary", "Symbol": "DHI"},{"Name": "Danaher Corp.", "Sector": "Health Care", "Symbol": "DHR"},{"Name": "Darden Restaurants", "Sector": "Consumer Discretionary", "Symbol": "DRI"},{"Name": "DaVita Inc.", "Sector": "Health Care", "Symbol": "DVA"},{"Name": "Deere & Co.", "Sector": "Industrials", "Symbol": "DE"},{"Name": "Delta Air Lines Inc.", "Sector": "Industrials", "Symbol": "DAL"},{"Name": "Dentsply Sirona", "Sector": "Health Care", "Symbol": "XRAY"},{"Name": "Devon Energy Corp.", "Sector": "Energy", "Symbol": "DVN"},{"Name": "Digital Realty Trust Inc", "Sector": "Real Estate", "Symbol": "DLR"},{"Name": "Discover Financial Services", "Sector": "Financials", "Symbol": "DFS"},{"Name": "Discovery Inc. Class A", "Sector": "Consumer Discretionary", "Symbol": "DISCA"},{"Name": "Discovery Inc. Class C", "Sector": "Consumer Discretionary", "Symbol": "DISCK"},{"Name": "Dish Network", "Sector": "Consumer Discretionary", "Symbol": "DISH"},{"Name": "Dollar General", "Sector": "Consumer Discretionary", "Symbol": "DG"},{"Name": "Dollar Tree", "Sector": "Consumer Discretionary", "Symbol": "DLTR"},{"Name": "Dominion Energy", "Sector": "Utilities", "Symbol": "D"},{"Name": "Dover Corp.", "Sector": "Industrials", "Symbol": "DOV"},{"Name": "DowDuPont", "Sector": "Materials", "Symbol": "DWDP"},{"Name": "Dr Pepper Snapple Group", "Sector": "Consumer Staples", "Symbol": "DPS"},{"Name": "DTE Energy Co.", "Sector": "Utilities", "Symbol": "DTE"},{"Name": "Duke Energy", "Sector": "Utilities", "Symbol": "DUK"},{"Name": "Duke Realty Corp", "Sector": "Real Estate", "Symbol": "DRE"},{"Name": "DXC Technology", "Sector": "Information Technology", "Symbol": "DXC"},{"Name": "E*Trade", "Sector": "Financials", "Symbol": "ETFC"},{"Name": "Eastman Chemical", "Sector": "Materials", "Symbol": "EMN"},{"Name": "Eaton Corporation", "Sector": "Industrials", "Symbol": "ETN"},{"Name": "eBay Inc.", "Sector": "Information Technology", "Symbol": "EBAY"},{"Name": "Ecolab Inc.", "Sector": "Materials", "Symbol": "ECL"},{"Name": "Edison Int'l", "Sector": "Utilities", "Symbol": "EIX"},{"Name": "Edwards Lifesciences", "Sector": "Health Care", "Symbol": "EW"},{"Name": "Electronic Arts", "Sector": "Information Technology", "Symbol": "EA"},{"Name": "Emerson Electric Company", "Sector": "Industrials", "Symbol": "EMR"},{"Name": "Entergy Corp.", "Sector": "Utilities", "Symbol": "ETR"},{"Name": "Envision Healthcare", "Sector": "Health Care", "Symbol": "EVHC"},{"Name": "EOG Resources", "Sector": "Energy", "Symbol": "EOG"},{"Name": "EQT Corporation", "Sector": "Energy", "Symbol": "EQT"},{"Name": "Equifax Inc.", "Sector": "Industrials", "Symbol": "EFX"},{"Name": "Equinix", "Sector": "Real Estate", "Symbol": "EQIX"},{"Name": "Equity Residential", "Sector": "Real Estate", "Symbol": "EQR"},{"Name": "Essex Property Trust, Inc.", "Sector": "Real Estate", "Symbol": "ESS"},{"Name": "Estee Lauder Cos.", "Sector": "Consumer Staples", "Symbol": "EL"},{"Name": "Everest Re Group Ltd.", "Sector": "Financials", "Symbol": "RE"},{"Name": "Eversource Energy", "Sector": "Utilities", "Symbol": "ES"},{"Name": "Exelon Corp.", "Sector": "Utilities", "Symbol": "EXC"},{"Name": "Expedia Inc.", "Sector": "Consumer Discretionary", "Symbol": "EXPE"},{"Name": "Expeditors International", "Sector": "Industrials", "Symbol": "EXPD"},{"Name": "Express Scripts", "Sector": "Health Care", "Symbol": "ESRX"},{"Name": "Extra Space Storage", "Sector": "Real Estate", "Symbol": "EXR"},{"Name": "Exxon Mobil Corp.", "Sector": "Energy", "Symbol": "XOM"},{"Name": "F5 Networks", "Sector": "Information Technology", "Symbol": "FFIV"},{"Name": "Facebook, Inc.", "Sector": "Information Technology", "Symbol": "FB"},{"Name": "Fastenal Co", "Sector": "Industrials", "Symbol": "FAST"},{"Name": "Federal Realty Investment Trust", "Sector": "Real Estate", "Symbol": "FRT"},{"Name": "FedEx Corporation", "Sector": "Industrials", "Symbol": "FDX"},{"Name": "Fidelity National Information Services", "Sector": "Information Technology", "Symbol": "FIS"},{"Name": "Fifth Third Bancorp", "Sector": "Financials", "Symbol": "FITB"},{"Name": "FirstEnergy Corp", "Sector": "Utilities", "Symbol": "FE"},{"Name": "Fiserv Inc", "Sector": "Information Technology", "Symbol": "FISV"},{"Name": "FLIR Systems", "Sector": "Information Technology", "Symbol": "FLIR"},{"Name": "Flowserve Corporation", "Sector": "Industrials", "Symbol": "FLS"},{"Name": "Fluor Corp.", "Sector": "Industrials", "Symbol": "FLR"},{"Name": "FMC Corporation", "Sector": "Materials", "Symbol": "FMC"},{"Name": "Foot Locker Inc", "Sector": "Consumer Discretionary", "Symbol": "FL"},{"Name": "Ford Motor", "Sector": "Consumer Discretionary", "Symbol": "F"},{"Name": "Fortive Corp", "Sector": "Industrials", "Symbol": "FTV"},{"Name": "Fortune Brands Home & Security", "Sector": "Industrials", "Symbol": "FBHS"},{"Name": "Franklin Resources", "Sector": "Financials", "Symbol": "BEN"},{"Name": "Freeport-McMoRan Inc.", "Sector": "Materials", "Symbol": "FCX"},{"Name": "Gap Inc.", "Sector": "Consumer Discretionary", "Symbol": "GPS"},{"Name": "Garmin Ltd.", "Sector": "Consumer Discretionary", "Symbol": "GRMN"},{"Name": "Gartner Inc", "Sector": "Information Technology", "Symbol": "IT"},{"Name": "General Dynamics", "Sector": "Industrials", "Symbol": "GD"},{"Name": "General Electric", "Sector": "Industrials", "Symbol": "GE"},{"Name": "General Growth Properties Inc.", "Sector": "Real Estate", "Symbol": "GGP"},{"Name": "General Mills", "Sector": "Consumer Staples", "Symbol": "GIS"},{"Name": "General Motors", "Sector": "Consumer Discretionary", "Symbol": "GM"},{"Name": "Genuine Parts", "Sector": "Consumer Discretionary", "Symbol": "GPC"},{"Name": "Gilead Sciences", "Sector": "Health Care", "Symbol": "GILD"},{"Name": "Global Payments Inc.", "Sector": "Information Technology", "Symbol": "GPN"},{"Name": "Goldman Sachs Group", "Sector": "Financials", "Symbol": "GS"},{"Name": "Goodyear Tire & Rubber", "Sector": "Consumer Discretionary", "Symbol": "GT"},{"Name": "Grainger (W.W.) Inc.", "Sector": "Industrials", "Symbol": "GWW"},{"Name": "Halliburton Co.", "Sector": "Energy", "Symbol": "HAL"},{"Name": "Hanesbrands Inc", "Sector": "Consumer Discretionary", "Symbol": "HBI"},{"Name": "Harley-Davidson", "Sector": "Consumer Discretionary", "Symbol": "HOG"},{"Name": "Harris Corporation", "Sector": "Information Technology", "Symbol": "HRS"},{"Name": "Hartford Financial Svc.Gp.", "Sector": "Financials", "Symbol": "HIG"},{"Name": "Hasbro Inc.", "Sector": "Consumer Discretionary", "Symbol": "HAS"},{"Name": "HCA Holdings", "Sector": "Health Care", "Symbol": "HCA"},{"Name": "HCP Inc.", "Sector": "Real Estate", "Symbol": "HCP"},{"Name": "Helmerich & Payne", "Sector": "Energy", "Symbol": "HP"},{"Name": "Henry Schein", "Sector": "Health Care", "Symbol": "HSIC"},{"Name": "Hess Corporation", "Sector": "Energy", "Symbol": "HES"},{"Name": "Hewlett Packard Enterprise", "Sector": "Information Technology", "Symbol": "HPE"},{"Name": "Hilton Worldwide Holdings Inc", "Sector": "Consumer Discretionary", "Symbol": "HLT"},{"Name": "Hologic", "Sector": "Health Care", "Symbol": "HOLX"},{"Name": "Home Depot", "Sector": "Consumer Discretionary", "Symbol": "HD"},{"Name": "Honeywell Int'l Inc.", "Sector": "Industrials", "Symbol": "HON"},{"Name": "Hormel Foods Corp.", "Sector": "Consumer Staples", "Symbol": "HRL"},{"Name": "Host Hotels & Resorts", "Sector": "Real Estate", "Symbol": "HST"},{"Name": "HP Inc.", "Sector": "Information Technology", "Symbol": "HPQ"},{"Name": "Humana Inc.", "Sector": "Health Care", "Symbol": "HUM"},{"Name": "Huntington Bancshares", "Sector": "Financials", "Symbol": "HBAN"},{"Name": "Huntington Ingalls Industries", "Sector": "Industrials", "Symbol": "HII"},{"Name": "IDEXX Laboratories", "Sector": "Health Care", "Symbol": "IDXX"},{"Name": "IHS Markit Ltd.", "Sector": "Industrials", "Symbol": "INFO"},{"Name": "Illinois Tool Works", "Sector": "Industrials", "Symbol": "ITW"},{"Name": "Illumina Inc", "Sector": "Health Care", "Symbol": "ILMN"},{"Name": "Incyte", "Sector": "Health Care", "Symbol": "INCY"},{"Name": "Ingersoll-Rand PLC", "Sector": "Industrials", "Symbol": "IR"},{"Name": "Intel Corp.", "Sector": "Information Technology", "Symbol": "INTC"},{"Name": "Intercontinental Exchange", "Sector": "Financials", "Symbol": "ICE"},{"Name": "International Business Machines", "Sector": "Information Technology", "Symbol": "IBM"},{"Name": "International Paper", "Sector": "Materials", "Symbol": "IP"},{"Name": "Interpublic Group", "Sector": "Consumer Discretionary", "Symbol": "IPG"},{"Name": "Intl Flavors & Fragrances", "Sector": "Materials", "Symbol": "IFF"},{"Name": "Intuit Inc.", "Sector": "Information Technology", "Symbol": "INTU"},{"Name": "Intuitive Surgical Inc.", "Sector": "Health Care", "Symbol": "ISRG"},{"Name": "Invesco Ltd.", "Sector": "Financials", "Symbol": "IVZ"},{"Name": "IPG Photonics Corp.", "Sector": "Information Technology", "Symbol": "IPGP"},{"Name": "IQVIA Holdings Inc.", "Sector": "Health Care", "Symbol": "IQV"},{"Name": "Iron Mountain Incorporated", "Sector": "Real Estate", "Symbol": "IRM"},{"Name": "J. B. Hunt Transport Services", "Sector": "Industrials", "Symbol": "JBHT"},{"Name": "Jacobs Engineering Group", "Sector": "Industrials", "Symbol": "JEC"},{"Name": "JM Smucker", "Sector": "Consumer Staples", "Symbol": "SJM"},{"Name": "Johnson & Johnson", "Sector": "Health Care", "Symbol": "JNJ"},{"Name": "Johnson Controls International", "Sector": "Industrials", "Symbol": "JCI"},{"Name": "JPMorgan Chase & Co.", "Sector": "Financials", "Symbol": "JPM"},{"Name": "Juniper Networks", "Sector": "Information Technology", "Symbol": "JNPR"},{"Name": "Kansas City Southern", "Sector": "Industrials", "Symbol": "KSU"},{"Name": "Kellogg Co.", "Sector": "Consumer Staples", "Symbol": "K"},{"Name": "KeyCorp", "Sector": "Financials", "Symbol": "KEY"},{"Name": "Kimberly-Clark", "Sector": "Consumer Staples", "Symbol": "KMB"},{"Name": "Kimco Realty", "Sector": "Real Estate", "Symbol": "KIM"},{"Name": "Kinder Morgan", "Sector": "Energy", "Symbol": "KMI"},{"Name": "KLA-Tencor Corp.", "Sector": "Information Technology", "Symbol": "KLAC"},{"Name": "Kohl's Corp.", "Sector": "Consumer Discretionary", "Symbol": "KSS"},{"Name": "Kraft Heinz Co", "Sector": "Consumer Staples", "Symbol": "KHC"},{"Name": "Kroger Co.", "Sector": "Consumer Staples", "Symbol": "KR"},{"Name": "L Brands Inc.", "Sector": "Consumer Discretionary", "Symbol": "LB"},{"Name": "L-3 Communications Holdings", "Sector": "Industrials", "Symbol": "LLL"},{"Name": "Laboratory Corp. of America Holding", "Sector": "Health Care", "Symbol": "LH"},{"Name": "Lam Research", "Sector": "Information Technology", "Symbol": "LRCX"},{"Name": "Leggett & Platt", "Sector": "Consumer Discretionary", "Symbol": "LEG"},{"Name": "Lennar Corp.", "Sector": "Consumer Discretionary", "Symbol": "LEN"},{"Name": "Leucadia National Corp.", "Sector": "Financials", "Symbol": "LUK"},{"Name": "Lilly (Eli) & Co.", "Sector": "Health Care", "Symbol": "LLY"},{"Name": "Lincoln National", "Sector": "Financials", "Symbol": "LNC"},{"Name": "LKQ Corporation", "Sector": "Consumer Discretionary", "Symbol": "LKQ"},{"Name": "Lockheed Martin Corp.", "Sector": "Industrials", "Symbol": "LMT"},{"Name": "Loews Corp.", "Sector": "Financials", "Symbol": "L"},{"Name": "Lowe's Cos.", "Sector": "Consumer Discretionary", "Symbol": "LOW"},{"Name": "LyondellBasell", "Sector": "Materials", "Symbol": "LYB"},{"Name": "M&T Bank Corp.", "Sector": "Financials", "Symbol": "MTB"},{"Name": "Macerich", "Sector": "Real Estate", "Symbol": "MAC"},{"Name": "Macy's Inc.", "Sector": "Consumer Discretionary", "Symbol": "M"},{"Name": "Marathon Oil Corp.", "Sector": "Energy", "Symbol": "MRO"},{"Name": "Marathon Petroleum", "Sector": "Energy", "Symbol": "MPC"},{"Name": "Marriott Int'l.", "Sector": "Consumer Discretionary", "Symbol": "MAR"},{"Name": "Marsh & McLennan", "Sector": "Financials", "Symbol": "MMC"},{"Name": "Martin Marietta Materials", "Sector": "Materials", "Symbol": "MLM"},{"Name": "Masco Corp.", "Sector": "Industrials", "Symbol": "MAS"},{"Name": "Mastercard Inc.", "Sector": "Information Technology", "Symbol": "MA"},{"Name": "Mattel Inc.", "Sector": "Consumer Discretionary", "Symbol": "MAT"},{"Name": "McCormick & Co.", "Sector": "Consumer Staples", "Symbol": "MKC"},{"Name": "McDonald's Corp.", "Sector": "Consumer Discretionary", "Symbol": "MCD"},{"Name": "McKesson Corp.", "Sector": "Health Care", "Symbol": "MCK"},{"Name": "Medtronic plc", "Sector": "Health Care", "Symbol": "MDT"},{"Name": "Merck & Co.", "Sector": "Health Care", "Symbol": "MRK"},{"Name": "MetLife Inc.", "Sector": "Financials", "Symbol": "MET"},{"Name": "Mettler Toledo", "Sector": "Health Care", "Symbol": "MTD"},{"Name": "MGM Resorts International", "Sector": "Consumer Discretionary", "Symbol": "MGM"},{"Name": "Michael Kors Holdings", "Sector": "Consumer Discretionary", "Symbol": "KORS"},{"Name": "Microchip Technology", "Sector": "Information Technology", "Symbol": "MCHP"},{"Name": "Micron Technology", "Sector": "Information Technology", "Symbol": "MU"},{"Name": "Microsoft Corp.", "Sector": "Information Technology", "Symbol": "MSFT"},{"Name": "Mid-America Apartments", "Sector": "Real Estate", "Symbol": "MAA"},{"Name": "Mohawk Industries", "Sector": "Consumer Discretionary", "Symbol": "MHK"},{"Name": "Molson Coors Brewing Company", "Sector": "Consumer Staples", "Symbol": "TAP"},{"Name": "Mondelez International", "Sector": "Consumer Staples", "Symbol": "MDLZ"},{"Name": "Monsanto Co.", "Sector": "Materials", "Symbol": "MON"},{"Name": "Monster Beverage", "Sector": "Consumer Staples", "Symbol": "MNST"},{"Name": "Moody's Corp", "Sector": "Financials", "Symbol": "MCO"},{"Name": "Morgan Stanley", "Sector": "Financials", "Symbol": "MS"},{"Name": "Motorola Solutions Inc.", "Sector": "Information Technology", "Symbol": "MSI"},{"Name": "Mylan N.V.", "Sector": "Health Care", "Symbol": "MYL"},{"Name": "Nasdaq, Inc.", "Sector": "Financials", "Symbol": "NDAQ"},{"Name": "National Oilwell Varco Inc.", "Sector": "Energy", "Symbol": "NOV"},{"Name": "Navient", "Sector": "Financials", "Symbol": "NAVI"},{"Name": "Nektar Therapeutics", "Sector": "Health Care", "Symbol": "NKTR"},{"Name": "NetApp", "Sector": "Information Technology", "Symbol": "NTAP"},{"Name": "Netflix Inc.", "Sector": "Information Technology", "Symbol": "NFLX"},{"Name": "Newell Brands", "Sector": "Consumer Discretionary", "Symbol": "NWL"},{"Name": "Newfield Exploration Co", "Sector": "Energy", "Symbol": "NFX"},{"Name": "Newmont Mining Corporation", "Sector": "Materials", "Symbol": "NEM"},{"Name": "News Corp. Class A", "Sector": "Consumer Discretionary", "Symbol": "NWSA"},{"Name": "News Corp. Class B", "Sector": "Consumer Discretionary", "Symbol": "NWS"},{"Name": "NextEra Energy", "Sector": "Utilities", "Symbol": "NEE"},{"Name": "Nielsen Holdings", "Sector": "Industrials", "Symbol": "NLSN"},{"Name": "Nike", "Sector": "Consumer Discretionary", "Symbol": "NKE"},{"Name": "NiSource Inc.", "Sector": "Utilities", "Symbol": "NI"},{"Name": "Noble Energy Inc", "Sector": "Energy", "Symbol": "NBL"},{"Name": "Nordstrom", "Sector": "Consumer Discretionary", "Symbol": "JWN"},{"Name": "Norfolk Southern Corp.", "Sector": "Industrials", "Symbol": "NSC"},{"Name": "Northern Trust Corp.", "Sector": "Financials", "Symbol": "NTRS"},{"Name": "Northrop Grumman Corp.", "Sector": "Industrials", "Symbol": "NOC"},{"Name": "Norwegian Cruise Line", "Sector": "Consumer Discretionary", "Symbol": "NCLH"},{"Name": "NRG Energy", "Sector": "Utilities", "Symbol": "NRG"},{"Name": "Nucor Corp.", "Sector": "Materials", "Symbol": "NUE"},{"Name": "Nvidia Corporation", "Sector": "Information Technology", "Symbol": "NVDA"},{"Name": "O'Reilly Automotive", "Sector": "Consumer Discretionary", "Symbol": "ORLY"},{"Name": "Occidental Petroleum", "Sector": "Energy", "Symbol": "OXY"},{"Name": "Omnicom Group", "Sector": "Consumer Discretionary", "Symbol": "OMC"},{"Name": "ONEOK", "Sector": "Energy", "Symbol": "OKE"},{"Name": "Oracle Corp.", "Sector": "Information Technology", "Symbol": "ORCL"},{"Name": "PACCAR Inc.", "Sector": "Industrials", "Symbol": "PCAR"},{"Name": "Packaging Corporation of America", "Sector": "Materials", "Symbol": "PKG"},{"Name": "Parker-Hannifin", "Sector": "Industrials", "Symbol": "PH"},{"Name": "Paychex Inc.", "Sector": "Information Technology", "Symbol": "PAYX"},{"Name": "PayPal", "Sector": "Information Technology", "Symbol": "PYPL"},{"Name": "Pentair Ltd.", "Sector": "Industrials", "Symbol": "PNR"},{"Name": "People's United Financial", "Sector": "Financials", "Symbol": "PBCT"},{"Name": "PepsiCo Inc.", "Sector": "Consumer Staples", "Symbol": "PEP"},{"Name": "PerkinElmer", "Sector": "Health Care", "Symbol": "PKI"},{"Name": "Perrigo", "Sector": "Health Care", "Symbol": "PRGO"},{"Name": "Pfizer Inc.", "Sector": "Health Care", "Symbol": "PFE"},{"Name": "PG&E Corp.", "Sector": "Utilities", "Symbol": "PCG"},{"Name": "Philip Morris International", "Sector": "Consumer Staples", "Symbol": "PM"},{"Name": "Phillips 66", "Sector": "Energy", "Symbol": "PSX"},{"Name": "Pinnacle West Capital", "Sector": "Utilities", "Symbol": "PNW"},{"Name": "Pioneer Natural Resources", "Sector": "Energy", "Symbol": "PXD"},{"Name": "PNC Financial Services", "Sector": "Financials", "Symbol": "PNC"},{"Name": "Polo Ralph Lauren Corp.", "Sector": "Consumer Discretionary", "Symbol": "RL"},{"Name": "PPG Industries", "Sector": "Materials", "Symbol": "PPG"},{"Name": "PPL Corp.", "Sector": "Utilities", "Symbol": "PPL"},{"Name": "Praxair Inc.", "Sector": "Materials", "Symbol": "PX"},{"Name": "Principal Financial Group", "Sector": "Financials", "Symbol": "PFG"},{"Name": "Procter & Gamble", "Sector": "Consumer Staples", "Symbol": "PG"},{"Name": "Progressive Corp.", "Sector": "Financials", "Symbol": "PGR"},{"Name": "Prologis", "Sector": "Real Estate", "Symbol": "PLD"},{"Name": "Prudential Financial", "Sector": "Financials", "Symbol": "PRU"},{"Name": "Public Serv. Enterprise Inc.", "Sector": "Utilities", "Symbol": "PEG"},{"Name": "Public Storage", "Sector": "Real Estate", "Symbol": "PSA"},{"Name": "Pulte Homes Inc.", "Sector": "Consumer Discretionary", "Symbol": "PHM"},{"Name": "PVH Corp.", "Sector": "Consumer Discretionary", "Symbol": "PVH"},{"Name": "Qorvo", "Sector": "Information Technology", "Symbol": "QRVO"},{"Name": "QUALCOMM Inc.", "Sector": "Information Technology", "Symbol": "QCOM"},{"Name": "Quanta Services Inc.", "Sector": "Industrials", "Symbol": "PWR"},{"Name": "Quest Diagnostics", "Sector": "Health Care", "Symbol": "DGX"},{"Name": "Range Resources Corp.", "Sector": "Energy", "Symbol": "RRC"},{"Name": "Raymond James Financial Inc.", "Sector": "Financials", "Symbol": "RJF"},{"Name": "Raytheon Co.", "Sector": "Industrials", "Symbol": "RTN"},{"Name": "Realty Income Corporation", "Sector": "Real Estate", "Symbol": "O"},{"Name": "Red Hat Inc.", "Sector": "Information Technology", "Symbol": "RHT"},{"Name": "Regency Centers Corporation", "Sector": "Real Estate", "Symbol": "REG"},{"Name": "Regeneron", "Sector": "Health Care", "Symbol": "REGN"},{"Name": "Regions Financial Corp.", "Sector": "Financials", "Symbol": "RF"},{"Name": "Republic Services Inc", "Sector": "Industrials", "Symbol": "RSG"},{"Name": "ResMed", "Sector": "Health Care", "Symbol": "RMD"},{"Name": "Robert Half International", "Sector": "Industrials", "Symbol": "RHI"},{"Name": "Rockwell Automation Inc.", "Sector": "Industrials", "Symbol": "ROK"},{"Name": "Rockwell Collins", "Sector": "Industrials", "Symbol": "COL"},{"Name": "Roper Technologies", "Sector": "Industrials", "Symbol": "ROP"},{"Name": "Ross Stores", "Sector": "Consumer Discretionary", "Symbol": "ROST"},{"Name": "Royal Caribbean Cruises Ltd", "Sector": "Consumer Discretionary", "Symbol": "RCL"},{"Name": "S&P Global, Inc.", "Sector": "Financials", "Symbol": "SPGI"},{"Name": "Salesforce.com", "Sector": "Information Technology", "Symbol": "CRM"},{"Name": "SBA Communications", "Sector": "Real Estate", "Symbol": "SBAC"},{"Name": "SCANA Corp", "Sector": "Utilities", "Symbol": "SCG"},{"Name": "Schlumberger Ltd.", "Sector": "Energy", "Symbol": "SLB"},{"Name": "Seagate Technology", "Sector": "Information Technology", "Symbol": "STX"},{"Name": "Sealed Air", "Sector": "Materials", "Symbol": "SEE"},{"Name": "Sempra Energy", "Sector": "Utilities", "Symbol": "SRE"},{"Name": "Sherwin-Williams", "Sector": "Materials", "Symbol": "SHW"},{"Name": "Simon Property Group Inc", "Sector": "Real Estate", "Symbol": "SPG"},{"Name": "Skyworks Solutions", "Sector": "Information Technology", "Symbol": "SWKS"},{"Name": "SL Green Realty", "Sector": "Real Estate", "Symbol": "SLG"},{"Name": "Snap-On Inc.", "Sector": "Consumer Discretionary", "Symbol": "SNA"},{"Name": "Southern Co.", "Sector": "Utilities", "Symbol": "SO"},{"Name": "Southwest Airlines", "Sector": "Industrials", "Symbol": "LUV"},{"Name": "Stanley Black & Decker", "Sector": "Consumer Discretionary", "Symbol": "SWK"},{"Name": "Starbucks Corp.", "Sector": "Consumer Discretionary", "Symbol": "SBUX"},{"Name": "State Street Corp.", "Sector": "Financials", "Symbol": "STT"},{"Name": "Stericycle Inc", "Sector": "Industrials", "Symbol": "SRCL"},{"Name": "Stryker Corp.", "Sector": "Health Care", "Symbol": "SYK"},{"Name": "SunTrust Banks", "Sector": "Financials", "Symbol": "STI"},{"Name": "SVB Financial", "Sector": "Financials", "Symbol": "SIVB"},{"Name": "Symantec Corp.", "Sector": "Information Technology", "Symbol": "SYMC"},{"Name": "Synchrony Financial", "Sector": "Financials", "Symbol": "SYF"},{"Name": "Synopsys Inc.", "Sector": "Information Technology", "Symbol": "SNPS"},{"Name": "Sysco Corp.", "Sector": "Consumer Staples", "Symbol": "SYY"},{"Name": "T. Rowe Price Group", "Sector": "Financials", "Symbol": "TROW"},{"Name": "Take-Two Interactive", "Sector": "Information Technology", "Symbol": "TTWO"},{"Name": "Tapestry, Inc.", "Sector": "Consumer Discretionary", "Symbol": "TPR"},{"Name": "Target Corp.", "Sector": "Consumer Discretionary", "Symbol": "TGT"},{"Name": "TE Connectivity Ltd.", "Sector": "Information Technology", "Symbol": "TEL"},{"Name": "TechnipFMC", "Sector": "Energy", "Symbol": "FTI"},{"Name": "Texas Instruments", "Sector": "Information Technology", "Symbol": "TXN"},{"Name": "Textron Inc.", "Sector": "Industrials", "Symbol": "TXT"},{"Name": "The Bank of New York Mellon Corp.", "Sector": "Financials", "Symbol": "BK"},{"Name": "The Clorox Company", "Sector": "Consumer Staples", "Symbol": "CLX"},{"Name": "The Cooper Companies", "Sector": "Health Care", "Symbol": "COO"},{"Name": "The Hershey Company", "Sector": "Consumer Staples", "Symbol": "HSY"},{"Name": "The Mosaic Company", "Sector": "Materials", "Symbol": "MOS"},{"Name": "The Travelers Companies Inc.", "Sector": "Financials", "Symbol": "TRV"},{"Name": "The Walt Disney Company", "Sector": "Consumer Discretionary", "Symbol": "DIS"},{"Name": "Thermo Fisher Scientific", "Sector": "Health Care", "Symbol": "TMO"},{"Name": "Tiffany & Co.", "Sector": "Consumer Discretionary", "Symbol": "TIF"},{"Name": "Time Warner Inc.", "Sector": "Consumer Discretionary", "Symbol": "TWX"},{"Name": "TJX Companies Inc.", "Sector": "Consumer Discretionary", "Symbol": "TJX"},{"Name": "Torchmark Corp.", "Sector": "Financials", "Symbol": "TMK"},{"Name": "Total System Services", "Sector": "Information Technology", "Symbol": "TSS"},{"Name": "Tractor Supply Company", "Sector": "Consumer Discretionary", "Symbol": "TSCO"},{"Name": "TransDigm Group", "Sector": "Industrials", "Symbol": "TDG"},{"Name": "TripAdvisor", "Sector": "Consumer Discretionary", "Symbol": "TRIP"},{"Name": "Twenty-First Century Fox Class A", "Sector": "Consumer Discretionary", "Symbol": "FOXA"},{"Name": "Twenty-First Century Fox Class B", "Sector": "Consumer Discretionary", "Symbol": "FOX"},{"Name": "Tyson Foods", "Sector": "Consumer Staples", "Symbol": "TSN"},{"Name": "U.S. Bancorp", "Sector": "Financials", "Symbol": "USB"},{"Name": "UDR Inc", "Sector": "Real Estate", "Symbol": "UDR"},{"Name": "Ulta Beauty", "Sector": "Consumer Discretionary", "Symbol": "ULTA"},{"Name": "Under Armour Class A", "Sector": "Consumer Discretionary", "Symbol": "UAA"},{"Name": "Under Armour Class C", "Sector": "Consumer Discretionary", "Symbol": "UA"},{"Name": "Union Pacific", "Sector": "Industrials", "Symbol": "UNP"},{"Name": "United Continental Holdings", "Sector": "Industrials", "Symbol": "UAL"},{"Name": "United Health Group Inc.", "Sector": "Health Care", "Symbol": "UNH"},{"Name": "United Parcel Service", "Sector": "Industrials", "Symbol": "UPS"},{"Name": "United Rentals, Inc.", "Sector": "Industrials", "Symbol": "URI"},{"Name": "United Technologies", "Sector": "Industrials", "Symbol": "UTX"},{"Name": "Universal Health Services, Inc.", "Sector": "Health Care", "Symbol": "UHS"},{"Name": "Unum Group", "Sector": "Financials", "Symbol": "UNM"},{"Name": "V.F. Corp.", "Sector": "Consumer Discretionary", "Symbol": "VFC"},{"Name": "Valero Energy", "Sector": "Energy", "Symbol": "VLO"},{"Name": "Varian Medical Systems", "Sector": "Health Care", "Symbol": "VAR"},{"Name": "Ventas Inc", "Sector": "Real Estate", "Symbol": "VTR"},{"Name": "Verisign Inc.", "Sector": "Information Technology", "Symbol": "VRSN"},{"Name": "Verisk Analytics", "Sector": "Industrials", "Symbol": "VRSK"},{"Name": "Verizon Communications", "Sector": "Telecommunication Services", "Symbol": "VZ"},{"Name": "Vertex Pharmaceuticals Inc", "Sector": "Health Care", "Symbol": "VRTX"},{"Name": "Viacom Inc.", "Sector": "Consumer Discretionary", "Symbol": "VIAB"},{"Name": "Visa Inc.", "Sector": "Information Technology", "Symbol": "V"},{"Name": "Vornado Realty Trust", "Sector": "Real Estate", "Symbol": "VNO"},{"Name": "Vulcan Materials", "Sector": "Materials", "Symbol": "VMC"},{"Name": "Wal-Mart Stores", "Sector": "Consumer Staples", "Symbol": "WMT"},{"Name": "Walgreens Boots Alliance", "Sector": "Consumer Staples", "Symbol": "WBA"},{"Name": "Waste Management Inc.", "Sector": "Industrials", "Symbol": "WM"},{"Name": "Waters Corporation", "Sector": "Health Care", "Symbol": "WAT"},{"Name": "Wec Energy Group Inc", "Sector": "Utilities", "Symbol": "WEC"},{"Name": "Wells Fargo", "Sector": "Financials", "Symbol": "WFC"},{"Name": "Welltower Inc.", "Sector": "Real Estate", "Symbol": "WELL"},{"Name": "Western Digital", "Sector": "Information Technology", "Symbol": "WDC"},{"Name": "Western Union Co", "Sector": "Information Technology", "Symbol": "WU"},{"Name": "WestRock Company", "Sector": "Materials", "Symbol": "WRK"},{"Name": "Weyerhaeuser Corp.", "Sector": "Real Estate", "Symbol": "WY"},{"Name": "Whirlpool Corp.", "Sector": "Consumer Discretionary", "Symbol": "WHR"},{"Name": "Williams Cos.", "Sector": "Energy", "Symbol": "WMB"},{"Name": "Willis Towers Watson", "Sector": "Financials", "Symbol": "WLTW"},{"Name": "Wyndham Worldwide", "Sector": "Consumer Discretionary", "Symbol": "WYN"},{"Name": "Wynn Resorts Ltd", "Sector": "Consumer Discretionary", "Symbol": "WYNN"},{"Name": "Xcel Energy Inc", "Sector": "Utilities", "Symbol": "XEL"},{"Name": "Xerox Corp.", "Sector": "Information Technology", "Symbol": "XRX"},{"Name": "Xilinx Inc", "Sector": "Information Technology", "Symbol": "XLNX"},{"Name": "XL Capital", "Sector": "Financials", "Symbol": "XL"},{"Name": "Xylem Inc.", "Sector": "Industrials", "Symbol": "XYL"},{"Name": "Yum! Brands Inc", "Sector": "Consumer Discretionary", "Symbol": "YUM"},{"Name": "Zimmer Biomet Holdings", "Sector": "Health Care", "Symbol": "ZBH"},{"Name": "Zions Bancorp", "Sector": "Financials", "Symbol": "ZION"},{"Name": "Zoetis", "Sector": "Health Care", "Symbol": "ZTS"}]      
  #sp500s = []
  obj_LSTM = s3.get_object(Bucket="lstm-returns-to-be-pulled-by-facebook-prophet", Key="Top 30 21-day Returns by LSTM.csv")
  sp500 = pd.read_csv(obj_LSTM['Body'])
  sp500s=sp500['Unnamed: 0'].tolist()
  
  #for entry in sp500:
  #    sp500s.append(entry["Symbol"])
  # sp500s = sp500s[:10] # debug
  for st in alreadyInvestedStocks:
      if st not in sp500s: # if not in there anyways
          sp500s.append(st)
  # need to add stocks we already invested in
  
  return sp500s

def get_data(stocks,alreadyInvestedStocks):
  print(type(stocks))
  if not type(stocks)==list:
        stocks=stocks.tolist()
  data = yf.download(stocks,period="1d",interval="1m")
  
  # get highest return stocks
  datan = data["Adj Close"].dropna(axis=1,how="all")
  datan = datan.dropna(axis=0,how="all")
  
  
  
  mean_returns = datan.pct_change()
  mean_returns = mean_returns.dropna(axis=0,how="all")
  mean_returns = mean_returns.replace([np.inf, -np.inf], 0)
  
  selection = mean_returns.mean().sort_values(ascending = False)[:50].index.to_list() # top 20 returns
  
  selection = np.concatenate([selection,alreadyInvestedStocks]) # add again already invested if not there already
  selection = np.unique(selection) # keep only unique of list
  print("length selection ",len(selection))
  datan = datan[selection]
  datan.index = pd.to_datetime(datan.index).tz_localize(None)
  return datan, mean_returns

def get_prediction(datan,STOCK):
  #print("START OF PREDICTION")
  #print(datan)
  showCharts = False
  daysOut = 15  #15

  stock_df = pd.DataFrame({
      'ds': datan.index,
      'y': datan[STOCK]
  })
  stock_df.dropna(inplace=True)
  # fit data using prophet model
  m = Prophet()
  m.fit(stock_df)

  # create future dates
  future_prices = m.make_future_dataframe(periods=daysOut, freq='min')

  # predict prices
  forecast = m.predict(future_prices)
  forecast['Stock']=STOCK
  cols='ds yhat yhat_lower yhat_upper Stock'
  
  

  # view results
  '''if showCharts:
      fig = m.plot(forecast)
      ax1 = fig.add_subplot(111)
      ax1.set_title("Stock Price Forecast", fontsize=16)
      ax1.set_xlabel("Date", fontsize=12)
      ax1.set_ylabel("Close Price", fontsize=12)

      fig2 = m.plot_components(forecast)
      plt.show()'''


  # calculate predicted returns
  end_of_period = datan.index[-1] + dt.timedelta(minutes=daysOut)

  # fix error
  forecast["ds"] = pd.to_datetime(forecast["ds"])
  end_of_period = pd.to_datetime(end_of_period)


  forecast = forecast.tail(daysOut)
  print("FORECASTS START HERE")
  #print(forecast)
  future_close_max = forecast.iloc[0].yhat_upper
  future_close_expected = forecast.iloc[0].yhat
  future_close_min = forecast.iloc[0].yhat_lower

  # calculate percent changes based on predictions
  max_move = (future_close_max - datan[STOCK][-1])/datan[STOCK][-1]
  expected_move = (future_close_expected - datan[STOCK][-1])/datan[STOCK][-1]
  min_move = (future_close_min - datan[STOCK][-1])/datan[STOCK][-1]
  #print("MOVES")
  print(max_move,expected_move,min_move)
  return max_move,expected_move,min_move

def trade_decision(invested,max_move,expected_move,min_move):
  # if the predicted movement is mostly positive, buy
  if max_move > 0 and abs(max_move) > abs(min_move):
      return "buy"

  # if the predicted movement is mostly negative, sell
  elif min_move < 0 and abs(min_move) > abs(max_move):
      
      # make sure we have some STOCK to sell
      if invested:
        return "sell"
  return None

# alpaca functions
def buy(api,stock,howmany,currentprice,investments):
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
    investments.update({stock:howmany})
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

def run(ALPACAUSER,ALPACAPW,GETNEWCONSTITUENTS = False, TRADE=False):
    api = tradeapi.REST(
        # TRADINGBOT1
        ALPACAUSER,
        ALPACAPW,
        'https://paper-api.alpaca.markets'
    )
    clock = api.get_clock()
    if not clock.is_open: # if markets open, else cancel
        account = api.get_account()
        buyingpower = float(account.buying_power) * 0.95 # to avoid overspending

      # get current investments
        portfolio = api.list_positions()
        portfolio_dict = {}
        for position in portfolio:
            portfolio_dict.update({str(position.symbol):int(position.qty)})
      # now get open orders, bc. we shouldnt do anything if its still unfullfilled
        openordersResp = api.list_orders(
          status='open',
          limit=100,
          nested=True  # show nested multi-leg orders
      )
        openorders = []
        for ord in openordersResp:
            openorders.append(ord.symbol)
        print("Open Orders, will skip: ",openorders)
  
      # rebalance the portfolio every monday, wednesday once per day. markets 
      # open at 9:30 a.m. ET to 4:00 p.m. ET (NYSE), which should be 
      # 13:30 - 20:00 in GMT, which lambda functions apparently use.
        now = dt.datetime.now()
        wd = now.weekday()
        current_time = now.strftime("%H:%M:%S")
        if wd in [0,2] and current_time < "13:40:00": # if it is monday or tuesday
            GETNEWCONSTITUENTS = True
            print("getting new constituents!")
      
        alreadyInvested = np.unique(list(portfolio_dict.keys()))
        GETNEWCONSTITUENTS=True
        if GETNEWCONSTITUENTS:
            constituents = get_constituents(alreadyInvested) # arg = already invested stocks (will be added)
        else:
            constituents = alreadyInvested # use already invested, should use the ones with highest return
      
      # remove open orders from constituents
        
        for stock in openorders:
            if stock in constituents:
              c=constituents.index(stock)
              constituents.pop(c)
            if stock in alreadyInvested:
              a=list(alreadyInvested).index(stock)
              alreadyInvested=np.delete(alreadyInvested, a)
            #constituents = np.delete(constituents, place)
            #placeTWO=np.where(constituents==stock)
            
            #alreadyInvested = np.delete(alreadyInvested, placeTWO)
        print(constituents)
        print("HELLEOEOEOE")
          # get the data from yfinance
        datan, df_stocks = get_data(constituents,alreadyInvested)
        
        constituents = datan.columns.to_list()
        print("DATAN")
        print(datan)
          # sometimes it happens that there still is an open order symbol in here
        for stock in openorders:
            constituents = np.delete(constituents, np.where(constituents == stock))

          # do new investing
        investments = {} # track current new investments
        decisions = {}
        Returns={}
        BuyDict={}
        for stock in constituents:
            
            if stock not in datan.columns.to_list():
                print("")
              #print(stock, " stock not there!!",datan.columns)
            else: # if data available
                print(f'{stock} IS PRESENT IN DATAN')
                max_move,expected_move,min_move = get_prediction(datan,stock)
                if stock in portfolio_dict:
                    invested = True
                else:
                    invested = False
                dec = trade_decision(invested,max_move,expected_move,min_move)
                decisions.update({stock:dec})
                Returns.update({stock: round(expected_move*100,2)})
                if dec=="buy":
                    BuyDict.update({stock:dec})
            
        nrbuys = len(list(decisions.values())[list(decisions.values())=="buy"])
        print("nr of planned buys",nrbuys)
        
        df_decisions=pd.DataFrame({'Stock': list(decisions.keys()), 'Decision': list(decisions.values())})
        df_returns=pd.DataFrame({'Stock': list(Returns.keys()), 'Return': list(Returns.values())})
        
        
        df_combined=pd.concat([df_decisions, df_returns.drop(columns='Stock')], axis=1)
        
        datan.to_csv("/tmp/Prophet-Full-Prices.csv")
        data = open('/tmp/Prophet-Full-Prices.csv', 'rb')
        s4.Bucket('facebook-prophet-layer').put_object(Key='Prophet-Full-Prices.csv', Body=data)
        
        
        df_combined.to_csv("/tmp/Prophet_Returns.csv")
        
        data = open('/tmp/Prophet_Returns.csv', 'rb')
        s4.Bucket('facebook-prophet-layer').put_object(Key='Prophet_Returns.csv', Body=data)
        
        df_dummy=pd.DataFrame({'A':5,
            'B':6}, index=['A', 'B'])
        df_dummy.to_csv("/tmp/dummy.csv")
        
        data=open('/tmp/dummy.csv', 'rb')
        s4.Bucket('dummy-2').put_object(Key='dummy.csv', Body=data)
        
        df_stocks[constituents].cov().to_csv("/tmp/Prophet_Covariance.csv")
        data = open('/tmp/Prophet_Covariance.csv', 'rb')
        s4.Bucket('facebook-prophet-layer').put_object(Key='Prophet_Covariance.csv', Body=data)
        
        '''df_returns.to_csv("/tmp/Prophet-Predicted-Returns.csv")
        
        data = open('/tmp/Prophet-Predicted-Returns.csv', 'rb')
        s4.Bucket('facebook-prophet-layer').put_object(Key='Prophet-Predicted-Returns.csv', Body=data)'''
        
        df_stocks[constituents].to_csv("/tmp/1-day-Prophet-Returns.csv")
        
        data = open('/tmp/1-day-Prophet-Returns.csv', 'rb')
        s4.Bucket('facebook-prophet-layer').put_object(Key='1-day-Prophet-Returns.csv', Body=data)
        
        #future_weights, backtest_returns=getWeightsReturns(df_stocks, df_returns)
        return datan, decisions
        
          # final buy
    else:
        return "Markets not open", ""

def lambda_handler(event, context):
   
    TRADE=False
    try:
        GETNEWCONSTITUENTS =True 
        #bool(os.getenv("C"))
        #(str(os.environ['GETNEWCONSTITUENTS']))
        ALPACAUSER =str(os.environ['ALPACAUSER'])
        ALPACAPW = str(os.environ['ALPACAPW'])
        datan, decisions=run(ALPACAUSER,ALPACAPW,True, TRADE)
        print("FINISHED")
        
        #print(datan)
        #print(decisions)
        message=f'Data Written successfully'
        return {
        "sessionAttributes": {
          "Full1DayPrices": json.loads(datan.iloc[[0]].to_json()),
          "Decisions": decisions
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
            
            
            
