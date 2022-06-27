import pandas as pd
import numpy as np
import requests

# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID
# 多空指標乖離
def Get_Stock_BBI(stock_code, start_date, stop_date): 
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    df = pd.json_normalize(result['data'])
    df = df.iloc[::-1]

    #計算M3、BS、M3-BS
    #分別計算3、6、9、12日均價後，再除以4計算多空指標
    #M3-BS數值可做為股票買賣時機參考
    df['M3'] = np.round(df['close'].rolling(window = 3).mean(), 2)
    df['M6'] = np.round(df['close'].rolling(window = 6).mean(), 2)
    df['M9'] = np.round(df['close'].rolling(window = 9).mean(), 2)
    df['M12'] = np.round(df['close'].rolling(window = 12).mean(), 2)
    df['BS'] = np.round((df['M3']+df['M6']+df['M9']+df['M12'])/4, 2)
    df['M3-BS'] = df['M3'] - df['BS']
    df = df.iloc[::-1]
    df = df.drop(['M6'], axis=1)
    df = df.drop(['M9'], axis=1)
    df = df.drop(['M12'], axis=1)
    js = df.to_json(orient='records')

    return js