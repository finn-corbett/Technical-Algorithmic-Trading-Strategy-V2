import sys
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
from datetime import date
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
sys.path.append(os.path.dirname(os.getcwd()))
path=os.getcwd() #get current directory
performance_filename=str(path)+"/performance.CSV" #get data filename
performance=pd.read_csv(performance_filename, skip_blank_lines=True, index_col=0)

Ticker_names=['AAPL', 'BRK.B', 'NEE', 'JPM', 'T', 'PG', 'WMT', 'VZ', 'WFC', 'BAC']
risk_free=3

algostd=[]
algo_yr_return=[]
n=[1-1, 253-1, 503-1, 755-1, 1007-1, 1258-1]
for x in range(1, len(performance)):
    algostd.append((sum([performance.iloc[x, n] for n in range(0,len(Ticker_names))])-sum([performance.iloc[x-1, n] for n in range(0,len(Ticker_names))]))*100/sum([performance.iloc[x-1, n] for n in range(0,len(Ticker_names))]))
for x in range(1, len(n)):
    val1=n[x]
    val2=n[x-1]
    algo_yr_return.append(100*(sum([performance.iloc[val1, n] for n in range(0,len(Ticker_names))])/sum([performance.iloc[val2, n] for n in range(0,len(Ticker_names))]))-100)

print('algo sharpe ratio;   ' + str((np.asarray(algo_yr_return).mean()-risk_free)/(np.asarray(algostd).std()*np.sqrt(252))))
bnhstd=[]
bnh_yr_return=[]
for x in range(1, len(performance)):
    bnhstd.append((sum([performance.iloc[x, n] for n in range(len(Ticker_names),len(Ticker_names)*2)])-sum([performance.iloc[x-1, n] for n in range(len(Ticker_names),len(Ticker_names)*2)]))*100/sum([performance.iloc[x-1, n] for n in range(len(Ticker_names),len(Ticker_names)*2)]))
for x in range(1, len(n)):
    val1=n[x]
    val2=n[x-1]
    bnh_yr_return.append(100*(sum([performance.iloc[val1, n] for n in range(len(Ticker_names),len(Ticker_names)*2)])/sum([performance.iloc[val2, n] for n in range(len(Ticker_names),len(Ticker_names)*2)]))-100)

print('buy and hold sharpe ratio:   ' + str((np.asarray(bnh_yr_return).mean()-risk_free)/(np.asarray(bnhstd).std()*np.sqrt(252))))

Tickers=['AAPL', 'BRK.B', 'NEE', 'JPM', 'T', 'PG', 'WMT', 'VZ', 'WFC', 'BAC']
paper_market="[insert paper markets key]"
key = "[insert data API key]"
secret = "[insert secret key]"
start="2017-09-01"
end="2022-09-01"

snp=[]

hsclient = StockHistoricalDataClient(key, secret)
quote_request_params=StockLatestQuoteRequest(symbol_or_symbols=Tickers)
dbar_request_params=StockBarsRequest(
    symbol_or_symbols=['SPY'],
    timeframe=TimeFrame.Day,
    start=start,
    end=end
    )
latest_quote=hsclient.get_stock_latest_quote(quote_request_params)
bars=hsclient.get_stock_bars(dbar_request_params)
data=bars.df
for i in range(0, len(data)-1):
    value=100000*data.iloc[i, 3]/data.iloc[0, 3]
    snp.append(value)

n=[1-1, 253-1, 503-1, 755-1, 1007-1, 1258-1]
snpannualreturns=[]
for x in range(1, len(n)):
    val1=n[x]
    val2=n[x-1]
    snpannualreturns.append((snp[val1]-snp[val2])*100/snp[val2])

snpreturns=[]

for i in range(1,len(snp)):
    snpreturns.append((snp[i]-snp[i-1])*100/snp[i-1])
print(np.asarray(snpannualreturns).mean())
print(np.asarray(snpreturns).std()*np.sqrt(252))

#PERFORMANCE MEASUREMENTS

#performance graphs for each stock
for q in range(0, len(Ticker_names)):
    plt.figure()
    plt.ylim(5000,50000)
    plt.title(Ticker_names[q]+' Trading Performance')
    plt.ylabel('Portfolio Value /$')
    plt.xlabel('Trading day')
    plt.plot(performance[Ticker_names[q]+' value'].to_list(), label=('algorithm'), color='k')
    plt.plot(performance[Ticker_names[q]+' buy and hold'].to_list(), label=('buy and hold'), color='orange')
    plt.plot(performance[Ticker_names[q]+' buy signals'].to_list(), label=('buy signals'), color='green', marker='o', linestyle='None')
    plt.plot(performance[Ticker_names[q]+' sell signals'].to_list(), label=('sell signals'), color='red', marker='o', linestyle='None')
    plt.legend()
    plt.show()

#overall performance graph
plt.figure()
plt.title('Overall Trading Performance')
plt.ylabel('Portfolio Value /$')
plt.xlabel('Trading day')
plt.plot(snp, label=('SPY'), color='c')
plt.plot([sum([performance.iloc[day, n] for n in range(len(Ticker_names),len(Ticker_names)*2)]) for day in range(0,len(performance))], label=('buy and hold'), color='orange')
plt.plot([sum([performance.iloc[day, n] for n in range(0,len(Ticker_names))]) for day in range(0,len(performance))], label=('value'), color='k')
plt.legend()
plt.show()

x=input('press [enter] key when done viewing figures to end program')
