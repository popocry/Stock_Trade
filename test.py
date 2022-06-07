import baostock as bs
import pandas as pd
import numpy as np


def get_data(stock_id='hz.600000', days_to_train=20, days_to_pred=5, start_data='2019-12-15', end_date='2020-12-15'):
    # 需要用20天的資料去預測未來五天的資料
    # days_to_train = 20
    # days_to_pred = 5

    # 登陸系統
    lg = bs.login()
    # 顯示登陸返回資訊
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 獲取滬深A股歷史K線資料
    # 引數說明：http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3#.E8.8E.B7.E5.8F.96.E5.8E.86.E5.8F.B2A.E8.82.A1K.E7.BA.BF.E6.95.B0.E6.8D.AE.EF.BC.9Aquery_history_k_data_plus.28.29
    rs = bs.query_history_k_data_plus(stock_id,
                                      "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                                      start_date=start_data, end_date=end_date,
                                      frequency="d", adjustflag="1")
    print('query_history_k_data_plus respond error_code:' + rs.error_code)
    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

    # 列印結果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 獲取一條記錄，將記錄合併在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    print(result)
    # 登出系統
    bs.logout()

    # 處理結果
    columns_all = result.columns
    columns_need = columns_all[2:-1]
    data_need = result[columns_need]
    column_low = 'low'
    column_high = 'high'

    # labels用於記錄股票在五天的時候是漲是跌
    # 漲：2
    # 平：1
    # 跌：0
    labels = []
    # train_data用於記錄上述分類中使用的訓練資料
    train_data = []
    for day in data_need.sort_index(ascending=False).index:
        day_pred_low = data_need.loc[day][column_low]
        day_pred_high = data_need.loc[day][column_high]
        if not (day - days_to_train - days_to_pred + 1 < 0):
            day_before_low = data_need.loc[day - days_to_pred][column_low]
            day_before_high = data_need.loc[day - days_to_pred][column_high]
            if day_pred_low > day_before_high:
                labels.append(2)
            elif day_pred_high < day_before_low:
                labels.append(0)
            else:
                labels.append(1)
            train_data.append(data_need.loc[day - days_to_pred - days_to_train + 1:day - days_to_pred])
    return train_data, labels

data_train, labels = get_data('sz.000651')
print(data_train, labels)