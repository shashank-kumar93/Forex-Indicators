import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def SMA(array, n):
    """Simple moving average"""
    return pd.Series(array).rolling(n).mean()

def EMA(ser, n):
    """Exponential moving average"""
    array = ser.copy()
    return pd.Series(array).ewm(span=n,adjust=False).mean()
    
def MACD(array,n):
    exp1 = EMA(pd.Series(array),12)
    exp2 = EMA(pd.Series(array),26)
    macd = exp1-exp2
    exp3 = EMA(pd.Series(macd),9)
    return exp3

def RSI(array, n):

    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)

def CCI(data_high, data_low, data_close, ndays): 
    TP = pd.Series((data_high + data_low + data_close) / 3 )
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
                    name = 'CCI') 
    return CCI


def Bollinger_Upper(array,n):
    SMA_20 = SMA(array,n)
    std_multiple = 2
    STD = pd.Series(array).rolling(n).std()
    return SMA_20 + (std_multiple*STD)

def Bollinger_Lower(array,n):
    SMA_20 = SMA(array,n)
    std_multiple = 2
    STD = pd.Series(array).rolling(n).std()
    return SMA_20 - (std_multiple*STD)

def ATR(high,low,n):
    atr = high-low
    return pd.Series(atr).rolling(n).mean()

def Donchain_High(data_high, n):
    data_high = pd.Series(data_high).rolling(n).max()
    return data_high

def Donchain_Low(data_low, n):
    data_low = pd.Series(data_low).rolling(n).min()
    return data_low

def Resistance_T1(close_p,open_p,low_p):
    dict_of_support = {}
    close_p = smooth(close_p)
    open_p = smooth(open_p)
    low_p = smooth(low_p)
    
    close_real = pd.Series(close_p)
    date_list_close = list(close_real.index)
    for i in date_list_close:
        dict_of_support[date_list_close[i]] = 0
    for i in range(len(close_p)-2):
        if min(abs(close_p.iloc[i] - open_p.iloc[i+1]), abs(low_p.iloc[i] - open_p.iloc[i+1])) <= 0.0001:
            if max(low_p.iloc[i], close_p.iloc[i]) < min(low_p.iloc[i+2],open_p.iloc[i+2]):
                dict_of_support[date_list_close[i]] = close_real.iloc[i]
    temp = 0
    for i in date_list_close:
        if dict_of_support[date_list_close[i]] != 0:
            temp = dict_of_support[date_list_close[i]]
        else:
            dict_of_support[date_list_close[i]] = temp

    support_series = pd.Series(dict_of_support, index = dict_of_support.keys())
    return support_series

def Support_T1(close_p,open_p,high_p):
    dict_of_resistance = {}
    close_p = smooth(close_p)
    open_p = smooth(open_p)
    high_p = smooth(high_p)
    close_real = pd.Series(close_p)
    date_list_close=list(close_real.index)
    for i in date_list_close:
        dict_of_resistance[date_list_close[i]]=0
    for i in range(len(close_p)-2):
        if min(abs(close_p.iloc[i] - open_p.iloc[i+1]), abs(high_p.iloc[i] - open_p.iloc[i+1])) <=0.0001:
            if min(high_p.iloc[i], close_p.iloc[i]) > max(high_p.iloc[i+2],open_p.iloc[i+2]):
                dict_of_resistance[date_list_close[i]] = close_real.iloc[i]
    temp = 0
    for i in date_list_close:
        if dict_of_resistance[date_list_close[i]]!=0:
            temp = dict_of_resistance[date_list_close[i]]
        else:
            dict_of_resistance[date_list_close[i]] = temp
    resistance_series = pd.Series(dict_of_resistance,index = dict_of_resistance.keys())
    return resistance_series

def Resistance_T2(close_p):
    
    close_series = pd.Series(close_p)
    close_np = np.asarray(close_p)
    close_np = np.round(close_np, decimals=4)
    array = np.diff(np.sign(np.diff(close_np)))
    array = np.insert(array, 0, 0)
    array = np.concatenate([array,[0]])
    date_list_close = list(close_series.index)
    dict_of_support = {}
    for i in date_list_close:
        dict_of_support[date_list_close[i]]=0
    for i in range(len(array)):
        if array[i]==-2:
            dict_of_support[date_list_close[i]]=close_series.iloc[i]
    
    temp = 0
    for i in date_list_close:
        if dict_of_support[date_list_close[i]]!=0:
            temp = dict_of_support[date_list_close[i]]
        else:
            dict_of_support[date_list_close[i]] = temp
    
    support_series = pd.Series(dict_of_support,index=dict_of_support.keys())
    return support_series

def Support_T2(close_p):
    
    close_series = pd.Series(close_p)
    close_np = np.asarray(close_p)
    close_np = np.round(close_np, decimals=4)
    dict_of_support = {}
    array = np.diff(np.sign(np.diff(close_np)))
    array = np.insert(array, 0, 0)
    array = np.concatenate([array,[0]])
    date_list_close = list(close_series.index)
    for i in date_list_close:
        dict_of_support[date_list_close[i]]=0
    
    for i in range(len(array)):
        if array[i]==2:
            dict_of_support[date_list_close[i]]=close_series.iloc[i]
    temp = 0
    for i in date_list_close:
        if dict_of_support[date_list_close[i]]!=0:
            temp = dict_of_support[date_list_close[i]]
        else:
            dict_of_support[date_list_close[i]] = temp

    support_series = pd.Series(dict_of_support,index=dict_of_support.keys())
    return support_series


def Support_T3(close_p):
    close_series = pd.Series(close_p)
    close_series = close_series.apply(lambda x : round(x,4))
    array = np.asarray(close_series)
    minm = argrelextrema(array, np.less)
    date_list_close = list(close_series.index)
    dict_of_support = {}
    for i in date_list_close:
        dict_of_support[date_list_close[i]]=0
    for i in minm[0]:
        dict_of_support[date_list_close[i]]=close_series.iloc[i]
    
    temp = 0
    for i in date_list_close:
        if dict_of_support[date_list_close[i]]!=0:
            temp = dict_of_support[date_list_close[i]]
        else:
            dict_of_support[date_list_close[i]] = temp
    
    support_series = pd.Series(dict_of_support,index=dict_of_support.keys())
    return support_series
    
def Resistance_T3(close_p):
    close_series = pd.Series(close_p)
    close_series = close_series.apply(lambda x : round(x,4))
    array = np.asarray(close_series)
    maxm = argrelextrema(array, np.greater)
    date_list_close = list(close_series.index)
    dict_of_support = {}
    for i in date_list_close:
        dict_of_support[date_list_close[i]]=0
    for i in maxm[0]:
        dict_of_support[date_list_close[i]]=close_series.iloc[i]
    
    temp = 0
    for i in date_list_close:
        if dict_of_support[date_list_close[i]]!=0:
            temp = dict_of_support[date_list_close[i]]
        else:
            dict_of_support[date_list_close[i]] = temp
    
    support_series = pd.Series(dict_of_support,index=dict_of_support.keys())
    return support_series

def Resistance_T4(close_p):
    dict_of_resistance = {}
    close_series = pd.Series(close_p)
    date_list_close = list(close_series.index)
    for i in date_list_close:
        dict_of_resistance[i] = 0
    close_p = np.array(close_p)
    close_p = smooth(close_p)
    array = np.diff(np.sign(np.diff(close_p)))
    array = np.insert(array, 0, 0)
    array = np.concatenate([array,[0]])
    for i in range(len(array)):
        if array[i]==-2:
            dict_of_resistance[date_list_close[i]]=close_series.iloc[i]
    temp = 0
    for i in date_list_close:
        if dict_of_resistance[i] != 0:
            temp = dict_of_resistance[i]
        else:
            dict_of_resistance[i] = temp

    resistance_series = pd.Series(dict_of_resistance, index = dict_of_resistance.keys())
    #resistance_series = pd.Series(dict_of_resistance)
    return resistance_series

def Support_T4(close_p):
    dict_of_support = {}
    close_series = pd.Series(close_p)
    date_list_close = list(close_series.index)
    for i in date_list_close:
        dict_of_support[i] = 0
    close_p = np.array(close_p)
    close_p = smooth(close_p)
    array = np.diff(np.sign(np.diff(close_p)))
    array = np.insert(array, 0, 0)
    array = np.concatenate([array,[0]])
    for i in range(len(array)):
        if array[i]==2:
            dict_of_support[date_list_close[i]] = close_series.iloc[i]
    temp = 0
    for i in range(len(date_list_close)):
        if dict_of_support[date_list_close[i]] != 0:
            temp = dict_of_support[date_list_close[i]]
        else:
            dict_of_support[date_list_close[i]] = temp

    support_series = pd.Series(dict_of_support, index = dict_of_support.keys())
    #support_series = pd.Series(dict_of_support)
    return support_series
def smooth(x):
    '''
    Smooth the data using a window with requested size.
    Adapted from:
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this:
    return y[(window_len/2-1):-(window_len/2)] instead of just y.
    '''
    window_len=11
    window='flat'
    if window_len < 3:  return x

    if x.ndim != 1: raise (StandardError('smooth only accepts 1 dimension arrays.'))
    if x.size < window_len:  raise (StandardError('Input vector needs to be bigger than window size.'))
    win_type = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    if window not in win_type: raise( StandardError( 'Window type is unknown'))

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    # saesha modify
    ds = y.shape[0] - x.shape[0] # difference of shape
    dsb = ds//2 # [almsot] half of the difference of shape for indexing at the begining
    dse = ds - dsb # rest of the difference of shape for indexing at the end
    y = y[dsb:-dse]

    return pd.Series(y)


