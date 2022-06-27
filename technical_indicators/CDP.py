import pandas as pd
import numpy as np
import requests

# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID
def Get_Stock_CDP(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    df = pd.json_normalize(result['data'])

    # 計算CDP、AH、NH、NL、AL
    # CDP = (前一日最高價 - 前一日最低價 + (前一日收盤價 * 2)) / 4
    # 最高值AH = CDP + ( 前一日最高價 - 前一日最低價 )
    # 近高值NH = CDP * 2 - 前一日最低價
    # 近低值NL = CDP * 2 - 前一日最高價
    # 最低值AL = CDP - ( 前一日最高價 - 前一日最低價 )
    df['CDP'] = (df["high"].shift(1) + df["low"].shift(1) + (df["close"].shift(1)) * 2) / 4
    df['AH'] = df['CDP'] +(df["high"].shift(1) - df["low"].shift(1))
    df['NH'] = df['CDP'] * 2 - df["low"].shift(1)
    df['NL'] = df['CDP'] * 2 - df["high"].shift(1)
    df['AL'] = df['CDP'] - (df["high"].shift(1) - df["low"].shift(1))
    js = df.to_json(orient='records')

    return js







