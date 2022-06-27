import pandas as pd
import numpy as np
import requests

# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID
#動向指標
def Get_Stock_DMI(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    df = pd.json_normalize(result['data'])
    df = df.iloc[::-1]
    
    #計算+DI、-DI、ADX
    def get_adx(high, low, close, period):
        P_dm = high.diff() #上升幅度
        N_dm = low.diff()  #下跌幅度
        P_dm[P_dm < 0] = 0
        N_dm[N_dm > 0] = 0
        
        h_l = pd.DataFrame(high - low) #當天的最高價－當天的最低價
        h_l2 = pd.DataFrame(abs(high - close.shift(1))) #當天的最高價－前一日的收盤價
        l_c = pd.DataFrame(abs(low - close.shift(1)))   #當天的最低價－前一日的收盤價
        frames = [h_l, h_l2, l_c]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1) #取最高值
        atr = tr.rolling(period).mean() #TR平均
        
        P_di = 100 * (P_dm.ewm(alpha = 1/period).mean() / atr)      #14日上升平均 / atr (+DI)
        N_di = abs(100 * (N_dm.ewm(alpha = 1/period).mean() / atr)) #14日下跌平均 / atr (-DI)
        dx = (abs(P_di - N_di) / abs(P_di + N_di)) * 100    #︱(+DI)-(-DI)︱/|(+DI)+(-DI)|
        adx = ((dx.shift(1) * (period - 1)) + dx) / period  #14日平均的移動平均值
        return P_di, N_di, adx

    df['+DI'] = pd.DataFrame(get_adx(df['high'], df['low'], df['close'], 14)[0]).rename(columns = {0:'P_di'})
    df['-DI'] = pd.DataFrame(get_adx(df['high'], df['low'], df['close'], 14)[1]).rename(columns = {0:'N_di'})
    df['ADX'] = pd.DataFrame(get_adx(df['high'], df['low'], df['close'], 14)[2]).rename(columns = {0:'adx'})
    df = df.iloc[::-1]
    js = df.to_json(orient='records')

    return js