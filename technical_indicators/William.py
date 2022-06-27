import pandas as pd
import numpy as np
import requests

# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID
#威廉指標
def Get_Stock_William(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    df = pd.json_normalize(result['data'])
    df = df.iloc[::-1]

    #計算William(W%R9)
    #100*(9日內最高價、最大值-當日收盤價)/(9日內最高價、最大值-9日內最低價、最小值)
    df['W%R9'] =  np.round(100 * (df['high'].rolling(9).max() - df['close'])
                / (df['high'].rolling(9).max() - df['low'].rolling(9).min()), 2)

    df = df.iloc[::-1]
    js = df.to_json(orient='records')

    return js

