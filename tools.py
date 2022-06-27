import requests
import json
import numpy as np
import time
import pandas as pd
import talib
from datetime import datetime, timedelta

def main(stock_name, start_date, target_date):
    # MA
    result, ma5_title = MA(stock_name, start_date, target_date, N=5)
    tmp, ma10_title = MA(stock_name, start_date, target_date, N=10)
    result = np.hstack((result, tmp[:, 1:2]))
    tmp, ma20_title = MA(stock_name, start_date, target_date, N=20)
    result = np.hstack((result, tmp[:, 1:2]))
    tmp, ma60_title = MA(stock_name, start_date, target_date, N=60)
    result = np.hstack((result, tmp[:, 1:2]))
    tmp, ma120_title = MA(stock_name, start_date, target_date, N=120)
    result = np.hstack((result, tmp[:, 1:2]))
    tmp, ma240_title = MA(stock_name, start_date, target_date, N=240)
    result = np.hstack((result, tmp[:, 1:2]))

    # KD
    tmp, [rsv, k9, d9] = KD(stock_name, start_date, target_date)
    result = np.hstack((result, tmp[:, 1:]))

    # MACD
    tmp, [macd, dif9, ema12, ema26] = MACD(stock_name, start_date, target_date)
    result = np.hstack((result, tmp[:, 1:]))

    # RSI
    tmp, rsi5 = RSI(stock_name, start_date, target_date, period=5)
    result = np.hstack((result, tmp[:, 1:]))
    tmp, rsi10 = RSI(stock_name, start_date, target_date, period=10)
    result = np.hstack((result, tmp[:, 1:]))

    # 乖離率
    tmp, bias10 = BIAS(stock_name, start_date, target_date, period=10)
    result = np.hstack((result, tmp[:, 1:]))
    tmp, bias20 = BIAS(stock_name, start_date, target_date, period=20)
    result = np.hstack((result, tmp[:, 1:]))

    # 威廉指數
    tmp, willr9 = WILLR(stock_name, start_date, target_date, period=9)
    result = np.hstack((result, tmp[:, 1:]))

    # CDP
    tmp, [cdp, ah, nh, nl, al] = CDP(stock_name, start_date, target_date)
    result = np.hstack((result, tmp[:, 1:]))

    titles = ['date', ma5_title, ma10_title, ma20_title, ma60_title,
                ma120_title, ma240_title, rsv, k9, d9,
                ema12, ema26, dif9, macd, rsi5, 
                rsi10, bias10, bias20, willr9, cdp,
                ah, nh, nl, al]
    json_list = []
    for i in range(result.shape[0]):
        # to timestamp
        result[i, 0] = time.mktime(result[i, 0].timetuple())
        json_list.append(dict(zip(titles, result[i])))
    
    return json.dumps({"data": json_list, 'title': titles}, indent=4)

def MA(stock_name, start_date, target_date, N):
    # heaader
    ma_name = "MA{}".format(N)
    # adjust start date 
    require_start_date = datetime.strptime(str(start_date), "%Y%m%d")
    if N == 3: require_start_date -= timedelta(days=10)
    elif N == 5: require_start_date -= timedelta(days=10)
    elif N == 6: require_start_date -= timedelta(days=20)
    elif N == 10: require_start_date -= timedelta(days=20)
    elif N == 12: require_start_date -= timedelta(days=20)
    elif N == 20: require_start_date -= timedelta(days=50)
    elif N == 24: require_start_date -= timedelta(days=50)
    elif N == 60: require_start_date -= timedelta(days=80)
    elif N == 120: require_start_date -= timedelta(days=150)
    elif N == 240: require_start_date -= timedelta(days=300)
    else:raise Exception("N must is only [3, 5, 6, 10, 12, 20, 24, 60, 120, 240]")
    require_start_date = require_start_date.strftime('%Y%m%d')

    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_name, require_start_date, target_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text)['data']
    history_data = [[datetime.fromtimestamp(int(data['date'])).date() ,data["close"]] for data in history_info]
    history_array = np.array(history_data)
    start_datetime = datetime.strptime(str(start_date), "%Y%m%d").date()
    # caculate MA
    need_index = np.argwhere(history_array[:, 0]>=start_datetime)
    ma_array = history_array[need_index].reshape(-1, 2)
    ma_array[:, 1] = 0

    for i in range(ma_array.shape[0]):
        ma_array[i, 1] = np.mean(history_array[i:i+N, 1])

    ma_array[:, 1] = np.round(ma_array[:, 1].astype(float), 2)

    return ma_array, ma_name

def KD(stock_name, start_date, target_date, init_K=47.44, init_D=36.64):
    # heaader
    kd_name = ['RSV', 'K9', 'D9']
    # adjust start date 
    require_start_date = datetime.strptime(str(start_date), "%Y%m%d")
    require_start_date -= timedelta(days=10)
    require_start_date = require_start_date.strftime('%Y%m%d')
    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_name, require_start_date, target_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text)['data']
    history_data = [[datetime.fromtimestamp(int(data['date'])).date(), data["high"], data["low"], data["close"]] for data in history_info]
    history_array = np.array(history_data)
    start_datetime = datetime.strptime(str(start_date), "%Y%m%d").date()
    # caculate KD
    need_index = np.argwhere(history_array[:, 0]>=start_datetime)
    kd_array = history_array[need_index].reshape(-1, 4)
    kd_array[:, 1:] = 0
    # RSV
    for i in range(kd_array.shape[0]):
        highest = history_array[i:i+9, 1].max()
        lowest = history_array[i:i+9, 2].min()
        kd_array[i, 1] = (history_array[i, 3] - lowest) / (highest-lowest) * 100
        
    # KD
    for i in range(kd_array.shape[0], -1, -1):
        if i==kd_array.shape[0]:
            # assign start_day-1's real K & real D
            pre_K, pre_D = init_K, init_D
        else:
            pre_rsv = kd_array[i, 1]
            # K
            kd_array[i, 2] = 2/3*pre_K + 1/3*pre_rsv
            # D
            kd_array[i, 3] = 2/3*pre_D + 1/3*kd_array[i, 2]
            # update K, D
            pre_K = kd_array[i, 2]
            pre_D = kd_array[i, 3]

    kd_array[:, 1:] = np.round(kd_array[:, 1:].astype(float), 2)
    
    return kd_array, kd_name
 
def MACD(stock_name, start_date, target_date):
    # heaader
    macd_name = ['MACD', 'DIF9', 'EMA12', 'EMA26']
    # adjust start date 
    require_start_date = datetime.strptime(str(start_date), "%Y%m%d")
    require_start_date -= timedelta(days=200)
    require_start_date = require_start_date.strftime('%Y%m%d')
    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_name, require_start_date, target_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text)['data']
    history_data = [[datetime.fromtimestamp(int(data['date'])).date(), data["close"]] for data in history_info]
    history_array = np.array(history_data)
    start_datetime = datetime.strptime(str(start_date), "%Y%m%d").date()

    # caculate MACD
    history_df = pd.DataFrame(history_data[::-1], columns=['date', 'close'])
    history_df['EMA12'] = history_df["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    history_df['EMA26'] = history_df["close"].ewm(span=26, adjust=False, min_periods=26).mean()
    history_df['DIF9']  = history_df['EMA12'] - history_df['EMA26']
    history_df['MACD'] = history_df['DIF9'].ewm(span=9, adjust=False).mean()
    # print(DIF)
    # history_df['MACD'] = 2*history_df['DIF9']-DIF
    need_index = np.argwhere(history_array[:, 0]>=start_datetime)
    # macd_array = history_array[need_index].reshape(-1, 2)

    macd_df = history_df.tail(need_index.shape[0])
    macd_df = macd_df.drop(columns=['close'])
    macd_array = macd_df.to_numpy()
    
    macd_array[:, 1:] = np.round(macd_array[:, 1:].astype(float), 2)
    
    return macd_array[::-1], macd_name

# RSI
def RSI(stock_name, start_date, target_date, period):
    # heaader
    result_name = 'RSI{}'.format(period)
    # adjust start date 
    require_start_date = datetime.strptime(str(start_date), "%Y%m%d")
    require_start_date -= timedelta(days=120)
    require_start_date = require_start_date.strftime('%Y%m%d')
    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_name, require_start_date, target_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text)['data']
    history_data = [[datetime.fromtimestamp(int(data['date'])).date(), data["close"]] for data in history_info]
    history_array = np.array(history_data)
    start_datetime = datetime.strptime(str(start_date), "%Y%m%d").date()

    history_df = pd.DataFrame(history_data[::-1], columns=['date', 'close'])
    
    # caculate RSI
    history_df['RSI'] = talib.RSI(history_df['close'], timeperiod=period)

    need_index = np.argwhere(history_array[:, 0]>=start_datetime)
    # macd_array = history_array[need_index].reshape(-1, 2)

    result_df = history_df.tail(need_index.shape[0])
    result_df = result_df.drop(columns=['close'])
    result = result_df.to_numpy()
    
    result[:, 1:] = np.round(result[:, 1:].astype(float), 2)

    return result[::-1], result_name

# 乖離率
def BIAS(stock_name, start_date, target_date, period):
    ma_result, ma = MA(stock_name, start_date, target_date, N=period)

    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_name, start_date, target_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text)['data']
    history_data = [[datetime.fromtimestamp(int(data['date'])).date(), data["close"]] for data in history_info]
    history_array = np.array(history_data)
    history_array[:, 1:] = (history_array[:, 1:] - ma_result[:, 1:])/ma_result[:, 1:]*100
    history_array[:, 1:] = np.round(history_array[:, 1:].astype(float), 2)

    return history_array, "BIAS{}".format(period)

# 威廉指數
def WILLR(stock_name, start_date, target_date, period):
    # heaader
    result_name = 'W%R{}'.format(period)
    # adjust start date 
    require_start_date = datetime.strptime(str(start_date), "%Y%m%d")
    require_start_date -= timedelta(days=120)
    require_start_date = require_start_date.strftime('%Y%m%d')
    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_name, require_start_date, target_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text)['data']
    history_data = [[datetime.fromtimestamp(int(data['date'])).date(), data["high"], data["low"], data["close"]] for data in history_info]
    history_array = np.array(history_data[::-1])
    # history_array[:, 1:] = history_array[:, 1:].astype(np.double)
    start_datetime = datetime.strptime(str(start_date), "%Y%m%d").date()

    history_df = pd.DataFrame(history_data[::-1], columns=['date', 'high', 'low', 'close'])
     # caculate WILLR
    history_df['willr'] = talib.WILLR(np.double(history_array[:, 1]), np.double(history_array[:, 2]), np.double(history_array[:, 3]), timeperiod=period)

    need_index = np.argwhere(history_array[:, 0]>=start_datetime)
    # macd_array = history_array[need_index].reshape(-1, 2)

    result_df = history_df.tail(need_index.shape[0])
    result_df = result_df.drop(columns=['close'])
    result_df = result_df.drop(columns=['high'])
    result_df = result_df.drop(columns=['low'])
    result = result_df.to_numpy()
    
    result[:, 1:] = np.round(-1*(result[:, 1:].astype(float)), 2)

    return result[::-1], result_name

# 多空指標乖離
def BBI(stock_name, start_date, target_date):
    # adjust start date 
    require_start_date = datetime.strptime(str(start_date), "%Y%m%d")
    require_start_date -= timedelta(days=120)
    require_start_date = require_start_date.strftime('%Y%m%d')

    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_name, require_start_date, target_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text)['data']
    history_data = [[datetime.fromtimestamp(int(data['date'])).date(), data["close"]] for data in history_info]
    history_array = np.array(history_data[::-1])

    ma3 = talib.MA(np.double(history_array[:, 1]),3)
    ma6 = talib.MA(np.double(history_array[:, 1]),6)
    ma12 = talib.MA(np.double(history_array[:, 1]),12)
    ma24 = talib.MA(np.double(history_array[:, 1]),24)
    # ma3, _ = MA(stock_name, start_date, target_date, N=3)
    # ma6, _ = MA(stock_name, start_date, target_date, N=6)
    # ma12, _ = MA(stock_name, start_date, target_date, N=12)
    # ma24, _ = MA(stock_name, start_date, target_date, N=20)
    print(ma12)

    print((ma3 + ma6 + ma12 + ma24)/4)
    # print((ma3[:, 1:] + ma6[:, 1:] + ma12[:, 1:] + ma24[:, 1:])/4)

# CDP
def CDP(stock_name, start_date, target_date):
    # heaader
    cdp_name = ['CDP', 'AH', 'NH', 'NL', 'AL']

    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_name, start_date, target_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text)['data']
    history_data = [[datetime.fromtimestamp(int(data['date'])).date(), data["low"], data["high"], data["close"]] for data in history_info]
    history_array = np.array(history_data[::-1])
    # pre_array = np.roll(history_array, 1, axis=0)
    # print(pre_array)
    cdp = (history_array[:, 1:2] + history_array[:, 2:3] + 2*history_array[:, 3:])/4
    ah = cdp + (history_array[:, 2:3] - history_array[:, 1:2])
    nh = 2*cdp  - history_array[:, 1:2]
    nl = 2*cdp  - history_array[:, 2:3]
    al = cdp - (history_array[:, 2:3] - history_array[:, 1:2])
    result = np.hstack((history_array[:, :1], cdp))
    result = np.hstack((result, ah))
    result = np.hstack((result, nh))
    result = np.hstack((result, nl))
    result = np.hstack((result, al))

    return result[::-1], cdp_name

def DMI(stock_name, start_date, target_date, di_period=7, adx_period=6):
    # adjust start date 
    require_start_date = datetime.strptime(str(start_date), "%Y%m%d")
    require_start_date -= timedelta(days=30)
    require_start_date = require_start_date.strftime('%Y%m%d')
    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_name, require_start_date, target_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text)['data']
    history_data = [[datetime.fromtimestamp(int(data['date'])).date(), data["high"], data["low"], data["close"]] for data in history_info]
    history_array = np.array(history_data[::-1])
    start_datetime = datetime.strptime(str(start_date), "%Y%m%d").date()
    need_index = np.argwhere(history_array[:, 0]>=start_datetime)

    # history_df = pd.DataFrame(history_data[::-1], columns=['date', 'high', 'low', 'close'])
    close = np.double(history_array[:, 3])
    high = np.double(history_array[:, 1])
    low = np.double(history_array[:, 2])

    pdi = talib.PLUS_DI(high, low, close, di_period) 
    mdi = talib.MINUS_DI(high, low, close, di_period)
    adx = talib.ADX(high, low, close, timeperiod=adx_period)
    adxr = talib.ADXR(high, low, close, timeperiod=adx_period)

    result = np.hstack((history_array[:, :1], pdi.reshape(-1, 1)))
    result = np.hstack((result, mdi.reshape(-1, 1)))
    result = np.hstack((result, adx.reshape(-1, 1)))
    result = np.hstack((result, adxr.reshape(-1, 1)))

    result[:, 1:] = np.round(-1*(result[:, 1:].astype(float)), 2)
    result = result[need_index]
    result = result.reshape(need_index.shape[0], 5)

    return result[::-1]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-stock',
                        '--stock_name',
                        type=int,
                        nargs='?',
                        help='Input stock name')

    parser.add_argument('-start',
                        '--start_date',
                        type=int,
                        nargs='?',
                        help='Input start date')

    parser.add_argument('-end',
                        '--end_date',
                        type=int,
                        nargs='?',
                        help='Input end date')

    args = parser.parse_args()
    json_result = main(args.stock_name, args.start_date, args.end_date)

    # json_result = main(2330, 20220421, 20220429)
    print(json_result)

    # json_result, _ = KD(2330, 20220421, 20220429)
    # print(json_result)
    # macd_array, _ = MACD(2330, 20220421, 20220429)
    # print(macd_array)
    # json_result = DMI(2330, 20220421, 20220429)
    # print(json_result)
