#a set of functions for calculating stock indicators

def calc_RSI(prices, period):
    if len(prices)<period+1:
        RSI=0
    else:
        gain=[]
        loss=[]
        for n in range(-period-1,0):
            if prices[n]>prices[n-1]:
                gain.append((prices[-1]/prices[-2]))
                loss.append(0)
            elif prices[n]<prices[n-1]:
                loss.append(abs(prices[-1]/prices[-2]))
                gain.append(0)
            else:
                gain.append(0)
                loss.append(0)
        if sum(loss)==0:
            RSI=100
        else:
            RSI=(100-(100/(1+(sum(gain))/(sum(loss)))))
    return RSI

def calc_SMA(prices, period):
    if len(prices)>=period:
        SMA=(sum(prices[-period-1:-1])/period)
    else:
        SMA=0
    return SMA
        
def calc_WMA(prices, period):
    price_vals=prices[-period-1:-1]
    if len(prices)>=period:
        WMA=sum([price_vals[n-1]*(n) for n in range(1, period+1)])/(period*(period+1)/2)
    else:
        WMA=0
    return WMA

def calc_EMA(prices, period, EMAs):
    k=2/(period+1)
    EMA=k*(prices[-1]-EMAs[-1])+EMAs[-1]
    return EMA

def calc_ROC(prices, period):
    ROC=100*(prices[-1]-prices[-1-period])/prices[-1-period]
    return ROC

def calc_CCI(prices, highs, lows, period):
    typical_price=(highs[-1]+lows[-1]+prices[-1])/3
    ma=(sum(prices[-period-1:-1])+sum(highs[-period-1:-1])+sum(lows[-period-1:-1]))/(3*period)
    deviation_vals=[]
    for n in range(-period-1,0):
        deviation_vals.append(abs(((highs[n]+lows[n]+prices[n])/3)-ma))
    deviation=sum(deviation_vals)
    CCI=(typical_price-ma)/(0.015*deviation)
    return CCI

def calc_R(prices, period):
    R=((max(prices[-period-1:-1])-prices[-1])/(max(prices[-period-1:-1])-min(prices[-period-1:-1])))
    return R

def calc_CMO(prices, period):
    higher_close=[]
    lower_close=[]
    for n in range(-period-1,0):
        if prices[n-1]<prices[n]:
            higher_close.append(prices[n])
        if prices[n-1]>prices[n]:
            lower_close.append(prices[n])
    CMO=100*(sum(higher_close)-sum(lower_close))/(sum(higher_close)+sum(lower_close))
    return CMO
