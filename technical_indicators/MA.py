import pandas as pd
import numpy as np
import requests

# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID
def Get_Stock_MA(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    df = pd.json_normalize(result['data'])
    
    #計算MA5、MA20、MA60
    #分別計算5、20、60日收盤均價
    df["MA5"] = np.round(df['close'].rolling(window=5).mean(), 2)
    df["MA20"] = np.round(df['close'].rolling(window=20).mean(), 2)
    df["MA60"] = np.round(df['close'].rolling(window=60).mean(), 2)
    js = df.to_json(orient='records')

    return js




