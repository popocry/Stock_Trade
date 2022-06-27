import pandas as pd
import numpy as np
import requests

# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID
RSI = []
def Get_Stock_RSI(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    df = pd.json_normalize(result['data'])
    df = df.iloc[::-1]

    #計算RSI5 =(5日)漲幅平均值 / ((5日)漲幅平均值＋(5日)日跌幅平均值)*100
    period_n = 5
    PrePrice = df['close'].diff()
    PrePriceValues = PrePrice.values #計算出收盤價差
    positive_differences = 0
    negative_differences = 0
    current_average_positive = 0
    current_average_negative = 0
    price_index = 0

    for i in range (period_n):
        RSI.append('NaN')

    for difference in PrePriceValues[1:]:
        if difference > 0:
            positive_difference = difference
            negative_difference = 0                
        if difference < 0:
            negative_difference = np.abs(difference)
            positive_difference = 0
        if difference == 0:
            negative_difference = 0
            positive_difference = 0

        #針對時間序列 < period_n(5日) 進行平均值初始化
        if (price_index < period_n):
            #計算(5日)漲幅平均、(5日)日跌幅平均
            current_average_positive = current_average_positive + (1 / period_n) * positive_difference
            current_average_negative = current_average_negative + (1 / period_n) * negative_difference
                
            if(price_index == (period_n - 1)):
                if current_average_negative != 0:
                    RSI.append(100 - 100 / (1+(current_average_positive / current_average_negative)))           
                else:
                    RSI.append(100)        
        #針對時間序列 > period_n(5日) 進行平均值遞歸更新
        else:
            #計算(5日)漲幅平均、(5日)日跌幅平均            
            current_average_positive = ((period_n-1) * current_average_positive+positive_difference) / (period_n)
            current_average_negative = ((period_n-1) * current_average_negative+negative_difference) / (period_n)
            
            if current_average_negative != 0:
                RSI.append(100 - 100 / (1+(current_average_positive / current_average_negative)))   
            else:
                RSI.append(100)
                
        price_index=price_index+1

    df['RSI5'] = RSI
    df = df.iloc[::-1]
    js = df.to_json(orient='records')

    return js