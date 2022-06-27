import pandas as pd
import numpy as np
import requests

# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID
def Get_Stock_MACD(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    
    result = requests.get(information_url).json()
    df = pd.json_normalize(result['data'])
    df = df.iloc[::-1]

    # 計算EMA12、EMA26、DIF9、MACD、OSC
    # 分別計算12、26日的指數移動平均值
    # 再以12日(短期)-26日(長期)取股價的差為DIF
    # 把9日的DIF再計算一次指數移動平均即為MACD
    # OSC(MACD Histogram)可做為判斷股票的買賣時機
    df['EMA12'] = np.round(df['close'].ewm(span=12).mean(), 2)
    df['EMA26'] = np.round(df['close'].ewm(span=26).mean(), 2)
    df['DIF9'] = np.round(df['EMA12'] - df['EMA26'], 2)
    df['MACD'] = np.round(df['DIF9'].ewm(span=9, adjust=False).mean(), 2)
    df['OSC'] = np.round(df['DIF9'] - df['MACD'], 2)
    df = df.iloc[::-1]
    js = df.to_json(orient='records')
    
    return js