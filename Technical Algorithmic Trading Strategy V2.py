from fileinput import close
from unicodedata import name
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
from datetime import date
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import math
import operator
import tensorflow.keras as tf
import indicators
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt; plt.ion()

# VARIABLE SETUP #
Ticker_variables=[]
data=[]

# TRADING CONFIG #
Ticker_names=['AAPL', 'BRK.B', 'NEE', 'JPM', 'T', 'PG', 'WMT', 'VZ', 'WFC', 'BAC']
paper_market="[insert paper market key here]"
key = "[insert data API key here]"
secret = "[insert secret key here]"

#"2015-12-15" earliest date
start="2017-09-01"
end="2022-09-01"

# PERFORMANCE CONFIG
risk_free=3

# NEURAL NETWORK CONFIG #
hidden_layer_count=3 #number of hidden layes in the neural network

# TRADING INDICATORS CONFIG #
#the periods used for calculating trading indicators are configured below
#new periods can be added or removed, the trading network will be dynamically adjusted to support more or fewer features
RSI_periods=[7,40]
MA_periods=[10,20] #Periods for SMA, WMA and EMA
MACD_EMA_periods=[[5,10,20],[10,20,40]] #EMA periods for MACD [signal, lower, higher]
ROC_periods=[10,20]
CCI_periods=[14,30]
R_periods=[14,30]
CMO_periods=[7,21]
#chaikin_periods=[[3,7],[10,16]]
#chaikin_storage=[[[], []] for n in range(0, len(chaikin_periods))]
wait_period=max(
    max(RSI_periods),
    max(MA_periods),
    max(max(MACD_EMA_periods[n]) for n in range(0,len(MACD_EMA_periods))),
    max(ROC_periods),
    max(R_periods),
    max(CCI_periods),
    max(CMO_periods)
)+1 #1 added to allow signals comparing current and yesterday values to be compared

features=len(RSI_periods)+len(MA_periods)*3+len(MACD_EMA_periods)+len(ROC_periods)+len(CCI_periods)+len(R_periods)+len(CMO_periods)

# machine learning signaller object classification
class signaller:
    def __init__(self):
        self.network=Sequential() #creates a sequential network

        #establish network size
        self.network.add(InputLayer(input_shape=(features, )))
        self.network.add(Dense(features*2, activation='sigmoid', name='1'))
        self.network.add(Dense(features*3, activation='sigmoid', name='2'))
        self.network.add(Dense(features*2, activation='sigmoid', name='3'))
        self.network.add(Dense(1))

        self.network.compile(
            optimizer=tf.optimizers.SGD(learning_rate=0.1),
            loss='MAE',
            )
    
    def predict(self, features):
        self.predictions=self.network.predict(features, batch_size=1, verbose=0)
        return self.predictions

    def reinforce(self, features, labels):
        self.network.fit(features, labels, epochs=1, batch_size=1, verbose=0, use_multiprocessing=True)

# ticker object classification, stores all data required for each ticker, with functions for calculating technical indicators
class ticker:
    def __init__(self, name):
        self.name=name
        self.close_prices=[]
        self.highs=[]
        self.lows=[]
        self.volume=[]
        self.EMA_storage=[[] for n in range(0, len(MA_periods))]
        self.MACD_storage=[[[], [], []] for n in range(0, len(MACD_EMA_periods))]
        self.cash=10000
        self.equity=0

    def new_data(self, price, high, low, volume):
        self.close_prices.append(price)
        self.highs.append(high)
        self.lows.append(low)
        self.volume.append(volume)

    #ALL CALCULATION FUNCTIONS USE THE MOST RECENT CLOSE_PRICE, DO NOT RUN THESE FUNCTIONS
    #BEFORE UPDATING USING ticker.new_price(price) 

    def RSI_signal(self, prices, period):
        if len(prices)<period:
            return 0.0
        else:
            RSI=indicators.calc_RSI(prices, period)
            return RSI/100

    def SMA_signal(self, prices, period):
        if len(prices)<period:
            return 0.0
        else:
            SMA=indicators.calc_SMA(prices, period)
            SMA_yesterday=indicators.calc_SMA(prices[0:-2], period)
            change=(SMA/SMA_yesterday)
            return change
    
    def WMA_signal(self, prices, period):
        if len(prices)<period:
            return 0.0
        else:
            WMA=indicators.calc_WMA(prices, period)
            WMA_yesterday=indicators.calc_WMA(prices[0:-2], period)
            change=(WMA/WMA_yesterday)
            return change
        
    def EMA_signal(self, prices, period, storage_val):
        if len(prices)<period:
            return 0.0
        else:
            if len(self.EMA_storage[storage_val])==0 or len(prices)==period:
                EMA=indicators.calc_SMA(prices, period)
                self.EMA_storage[storage_val].append(EMA)
                return 0.0
            else:
                EMA=indicators.calc_EMA(prices, period, self.EMA_storage[storage_val])
                self.EMA_storage[storage_val].append(EMA)
                change=(self.EMA_storage[storage_val][-1]/self.EMA_storage[storage_val][-2])
                return change

    def MACD_signal(self, prices, periods, storage_val):
        if len(prices)<max(periods):
            return 0.0
        elif len(prices)==max(periods) or len(self.MACD_storage[storage_val][0])==0:
            signal_EMA=indicators.calc_SMA(prices, periods[0])
            low_EMA=indicators.calc_SMA(prices, periods[1])
            high_EMA=indicators.calc_SMA(prices, periods[2])
            self.MACD_storage[storage_val][0].append(signal_EMA)
            self.MACD_storage[storage_val][1].append(low_EMA)
            self.MACD_storage[storage_val][2].append(high_EMA)
            if signal_EMA>(low_EMA-high_EMA):
                return 1.0
            else:
                return 0.0
        else:
            signal_EMA=indicators.calc_EMA(prices, periods[0], self.MACD_storage[storage_val][0])
            low_EMA=indicators.calc_EMA(prices, periods[1], self.MACD_storage[storage_val][1])
            high_EMA=indicators.calc_EMA(prices, periods[2], self.MACD_storage[storage_val][2])
            if signal_EMA>(low_EMA-high_EMA):
                return 1.0
            else:
                return 0.0

    def ROC_signal(self, prices, period):
        if len(prices)<period:
            return 0.0
        else:
            ROC=indicators.calc_ROC(prices, period)
            if ROC>0:
                return 1.0
            else:
                return 0.0

    def CCI_signal(self, prices, highs, lows, period):
        if len(prices)<period:
            return 0.0
        else:
            CCI=indicators.calc_CCI(prices, highs, lows, period)
            return CCI/100

    def R_signal(self, prices, period):
        if len(prices)>=period:
            R=indicators.calc_R(prices, period)
        return (abs(R)/100)
    
    def CMO_signal(self, prices, period):
        if len(prices)<period:
            return 0.0
        else:
            CMO=indicators.calc_CMO(prices, period)
            return ((CMO+100)/200)

    def calc_signals(self):
        RSI_vals=[]
        for n in range(0,len(RSI_periods)):
            RSI_vals.append(self.RSI_signal(self.close_prices, RSI_periods[n]))
        SMA_vals=[]
        for n in range(0,len(MA_periods)):
            SMA_vals.append(self.SMA_signal(self.close_prices, MA_periods[n]))
        WMA_vals=[]
        for n in range(0,len(MA_periods)):
            WMA_vals.append(self.WMA_signal(self.close_prices, MA_periods[n]))
        EMA_vals=[]
        for n in range(0,len(MA_periods)):
            EMA_vals.append(self.EMA_signal(self.close_prices, MA_periods[n], n))
        MACD_vals=[]
        for n in range(0,len(MACD_EMA_periods)):
            MACD_vals.append(self.MACD_signal(self.close_prices, MACD_EMA_periods[n], n))
        ROC_vals=[]
        for n in range(0,len(ROC_periods)):
            ROC_vals.append(self.ROC_signal(self.close_prices, ROC_periods[n]))
        CCI_vals=[]
        for n in range(0,len(CCI_periods)):
            CCI_vals.append(self.CCI_signal(self.close_prices, self.highs, self.lows, CCI_periods[n]))
        R_vals=[]
        for n in range(0,len(R_periods)):
            R_vals.append(self.R_signal(self.close_prices, R_periods[n]))
        CMO_vals=[]
        for n in range(0,len(CMO_periods)):
            CMO_vals.append(self.CMO_signal(self.close_prices, CMO_periods[n]))
        signals=[RSI_vals+SMA_vals+WMA_vals+EMA_vals+MACD_vals+ROC_vals+CCI_vals+R_vals+CMO_vals]
        return signals

# GET PRICE DATA #
# collect full historical data for each ticker,
# will not be required in live version
# (may be used to calculate rolling indicators for immediate trading)
for q in range(0, len(Ticker_names)):
    hsclient = StockHistoricalDataClient(key, secret)
    dbar_request_params=StockBarsRequest(
        symbol_or_symbols=Ticker_names[q],
        timeframe=TimeFrame.Day,
        start=start,
        end=end
        )
    bars=hsclient.get_stock_bars(dbar_request_params)
    hsclient.get
    data.append(bars.df)
# function that waits until a price update before storing the prices for each ticker in a variable
def get_daily_data(day, q):
    daily_close_price=data[q].iloc[day, 3]
    daily_high=data[q].iloc[day, 1]
    daily_low=data[q].iloc[day, 2]
    daily_volume=data[q].iloc[day, 4]

    #rules for stocks splits for NEE and AAPL (adjusts price to ignore the stock split, stocks are not multiplied in code)
    if (q==2 and day>=793) or (q==0 and day>=753):
        daily_close_price*=4
        daily_high*=4
        daily_low*=4
    return daily_close_price, daily_high, daily_low, daily_volume

performance_columns=[n+' value' for n in Ticker_names] + [n+' buy and hold' for n in Ticker_names] + [n+' buy signals' for n in Ticker_names] + [n+' sell signals' for n in Ticker_names]
performance=pd.DataFrame(columns=performance_columns)

for q in range(0, len(Ticker_names)):
    Ticker_variables.append(ticker(Ticker_names[q]))

# MAIN CODE BODY #
network=signaller() #define the neural network variable

for day in range(0, len(data[0])-1):
    performance=performance.append(pd.Series(0, index=performance.columns), ignore_index=True) #add new line to the data dataframe to input data for the new trading day
    for q in range(0, len(Ticker_names)):
        Ticker_variables[q].new_data(get_daily_data(day, q)[0], get_daily_data(day, q)[1], get_daily_data(day, q)[2], get_daily_data(day, q)[3]) #calculate indicators for new data

        #calculate buy/sell signals
        buy_signal=0
        sell_signal=0
        if day>wait_period:
            signal=network.predict(Ticker_variables[q].calc_signals())
            if signal[0][0]>0: #signal to buy
                if Ticker_variables[q].cash>0:
                    buy_signal=Ticker_variables[q].cash+(Ticker_variables[q].equity*Ticker_variables[q].close_prices[-1])

                Ticker_variables[q].equity+=Ticker_variables[q].cash/Ticker_variables[q].close_prices[-1]
                Ticker_variables[q].cash=0
                sell_signal=0

            else: # signal to sell
                if Ticker_variables[q].equity>0:
                    sell_signal=Ticker_variables[q].cash+(Ticker_variables[q].equity*Ticker_variables[q].close_prices[-1])

                Ticker_variables[q].cash+=Ticker_variables[q].equity*Ticker_variables[q].close_prices[-1]
                Ticker_variables[q].equity=0
                buy_signal=0
                
        #append performance metrics for each ticker
        performance.iat[day, q]=Ticker_variables[q].cash+(Ticker_variables[q].equity*Ticker_variables[q].close_prices[-1])
        performance.iat[day, q+len(Ticker_names)]=10000*(Ticker_variables[q].close_prices[-1]/Ticker_variables[q].close_prices[0])
        performance.iat[day, q+2*len(Ticker_names)]=buy_signal
        performance.iat[day, q+3*len(Ticker_names)]=sell_signal
        
    #reinforce neural network
    for q in range(0, len(Ticker_names)):
        if day>wait_period:
            Ticker_variables[q].new_data(get_daily_data(day, q)[0], get_daily_data(day, q)[1], get_daily_data(day, q)[2], get_daily_data(day, q)[3])
            label=[((data[q].iloc[day+1, 3])-(data[q].iloc[day, 3]))/(data[q].iloc[day, 3])*100]
            signal=network.reinforce(Ticker_variables[q].calc_signals(), label)

    #prints the value of the portfolio as the code runs
    portfolio_value=sum([Ticker_variables[q].cash+(Ticker_variables[q].equity*Ticker_variables[q].close_prices[-1]) for q in range(0, len(Ticker_variables))])
    print(portfolio_value)

x=input('press [enter] for detailed performance metrics')

#PERFORMANCE MEASUREMENTS

#performance graphs for each stock
for q in range(0, len(Ticker_names)):
    plt.figure()
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
plt.plot([sum([performance.iloc[day, n] for n in range(0,len(Ticker_names))]) for day in range(0,len(data[0])-1)], label=('value'), color='k')
plt.plot([sum([performance.iloc[day, n] for n in range(len(Ticker_names),len(Ticker_names)*2)]) for day in range(0,len(data[0])-1)], label=('buy and hold'), color='orange')
plt.legend()
plt.show()

performance.to_csv('performance.CSV') #performance CSV used with performance_eval.py to display detailed performance
x=input('press [enter] key when done viewing figures to end program')
