import pandas as pd
import numpy as np
import requests

# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID
low_list=[]
high_lost=[]
def Get_Stock_KDJ(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    df = pd.json_normalize(result['data'])
    df = df.iloc[::-1]

    #計算RSV、K9、D9、J9、3K-2D
    #先計算9日內最低價(low_list)與最高價(high_list)
    #接著計算RSV = (當天收盤價 - 9日內最低價) / (9日內最高價 - 9日內最高價)
    #K值=(2/3)*前一日K值 + (1/3)*當日RSV(以遞歸方式計算加權平均值)，D值同理
    #J值=3*D-2*K(3K-2D值同理)
    low_list = np.round(df['low'].rolling(9, min_periods=9).min(), 2)
    low_list.fillna(value=df['low'].expanding().min(), inplace=True)
    high_list = np.round(df['high'].rolling(9, min_periods=9).max(), 2)
    high_list.fillna(value=df['high'].expanding().max(), inplace=True)
    df['RSV'] = np.round((df['close'] - low_list) / (high_list - low_list) * 100, 2)
    df['K9'] = np.round(df['RSV'].ewm(com=2).mean(), 2)
    df['D9'] = np.round(df['K9'].ewm(com=2).mean(), 2)
    df['J9'] = np.round(3 * df['D9'] - 2 * df['K9'], 2)
    df['3K-2D'] = np.round(3 * df['K9'] - 2 * df['D9'], 2)
    df = df.iloc[::-1]
    js = df.to_json(orient='records')

    return js





