import pandas as pd
import numpy as np
import requests

# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID
# 乖離率
def Get_Stock_BIAS(stock_code, start_date, stop_date): 
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    df = pd.json_normalize(result['data'])
    df = df.iloc[::-1]

    #計算BIAS10、BIAS20、B10-B20
    #BIAS = (當日收盤價 – N日移動平均價) ÷ N日移動平均價 x 100
    df['BIAS10'] = np.round((df['close'] - df['close'].rolling(10).mean())
                            / df['close'].rolling(10).mean()*100, 2)
    df['BIAS20'] = np.round((df['close'] - df['close'].rolling(20).mean())
                            / df['close'].rolling(20).mean()*100, 2)
    df['B10-B20'] = df['BIAS10'] - df['BIAS20']
    df = df.iloc[::-1]
    js = df.to_json(orient='records')

    return js





