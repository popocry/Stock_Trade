import requests,json
from io import StringIO
import pandas as pd
import datetime
import configparser
from pandas import json_normalize
from sklearn import preprocessing
from GRUNet import Train_model, build_model
import os
import numpy as np
import torch


class stock_trade:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('setting.ini')
        self.account = config['setup']['id']#帳號
        self.password = config['setup']['password']#密碼
        self.start_date = "19701015"#開始日期
        self.end_date = datetime.datetime.now().strftime('%Y%m%d')#結束日期
        self.top = 50 #前top買賣超
        self.days_to_train = 20
        self.days_to_pred = 3
        self.model = build_model()
    
    def Norm_Data(self, data):
        df = pd.DataFrame(data)
        df = df.drop(columns = ['capacity', 'date', 'stock_code_id'])
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))#縮放至0-1
        scaled_data = scaler.fit_transform(df)
        return scaled_data

    def Stock_Movement_Data(self, stock_information):
        # labels用於記錄股票在days_to_pred天後的時候是漲是跌
        # 漲：2
        # 平：1
        # 跌：0
        labels = []
        # dataset用於記錄上述分類中使用的訓練資料
        dataset = []
        stock_train_information = self.Norm_Data(stock_information)# 只取訓練所需資料
        stock_information = stock_information[::-1]# 由舊到新排序
        for day, oneday_stock_information in enumerate(stock_information):# 從最早開始遍歷股票資料
            day_pred_low = oneday_stock_information['low']# 未來第days_to_pred天最低價格
            day_pred_high = oneday_stock_information['high']# 未來第days_to_pred天最高價格

            # 建立訓練資料
            if (day - self.days_to_train - self.days_to_pred + 1 >=0):
                day_before_low = stock_information[day - self.days_to_pred]['low']# days_to_pred天前股價的最低價格
                day_before_high = stock_information[day - self.days_to_pred]['high']# days_to_pred天前股價的最高價格
                if day_pred_low > day_before_high:# 漲
                    labels.append(2)
                elif day_pred_high < day_before_low:# 跌
                    labels.append(0)
                else:# 持平
                    labels.append(1)
                dataset.append(stock_train_information[day - self.days_to_pred - self.days_to_train + 1:day - self.days_to_pred + 1])
        return dataset, labels

    def Stock_Pred_Data(self, stock_information):
        stock_pred_data = self.Norm_Data(stock_information)# 正規化
        stock_pred_data = stock_pred_data[::-1]# 由舊到新排序
        stock_pred_data = stock_pred_data[-self.days_to_train:]
        stock_pred_data = stock_pred_data.reshape((1, stock_pred_data.shape[0], stock_pred_data.shape[1]))
        return stock_pred_data

    def Pred_Stock_Movement(self, stock_code, data):
        self.model.load_state_dict(torch.load('./model/best' + stock_code + '.pth'))
        self.model.eval()
        data = torch.from_numpy(np.flip(data,axis=0).copy())
        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)# 輸出最大值索引位置
            predicted = predicted.numpy()[0]
            softmax = torch.nn.Softmax(dim =1)# Softmax
            outputs = softmax(outputs).numpy().reshape(-1)# softmax轉成機率
        return predicted, outputs[predicted]

    def Buy_Stock(self, account, password, stock_code, stock_shares, stock_price):
        data = data = {'account':account, 'password':password, 'stock_code':stock_code, 'stock_shares':stock_shares, 'stock_price':stock_price}
        sell_url = 'http://140.116.86.242:8081/stock/api/v1/buy'
        result = requests.post(sell_url, data = data).json()
        if(result['result'] == 'success'):
            print("股票代號: {} 以每張 : {} 買入: {}張".format(stock_code, stock_price, stock_shares))
        return result['result'] == 'success'

    def Buy(self):
        date = datetime.datetime.now().strftime('%Y%m%d')
        date = "20220606"
        r = requests.get('http://www.tse.com.tw/fund/T86?response=csv&date='+date+'&selectType=ALLBUT0999')# 獲取三大法人買賣超日報
        df = pd.read_csv(StringIO(r.text), header=1).dropna(how='all', axis=1).dropna(how='any')
        for i, stock_code in enumerate(df['證券代號'].values):
            if '=' in stock_code:# 刪除系統沒有的股票
                df.drop(i, axis=0, inplace=True)
        
        for stock_code in df[0:self.top]['證券代號']:# 獲取前top買賣超股票代號
            path = os.path.abspath(os.getcwd())
            model_path = path + '/model/best'+ stock_code + '.pth'

            # 判斷模型是否存在，模型不存在的話進行訓練
            if not os.path.exists(model_path):
                stock_information = self.Get_Stock_Information(stock_code, self.start_date, self.end_date)# 獲取股票資訊
                if stock_information == {}:# 處理不在系統的股票
                    continue
                dataset, labels = self.Stock_Movement_Data(stock_information)# 訓練格式
                Train_model(stock_code, dataset, labels)# 訓練

            # 對未來股票趨勢預測
            stock_information = self.Get_Stock_Information(stock_code, self.start_date, self.end_date)# 獲取股票資訊
            if stock_information == {}:# 處理不在系統的股票
                continue
            data = self.Stock_Pred_Data(stock_information)# 將股票資料轉為預測格式
            pred, accuracy = self.Pred_Stock_Movement(stock_code, data)# 預測未來股票趨勢(準確率)
            if pred == 2:# 未來看漲買入
                if accuracy < 0.5:
                    buy_shares = 3
                elif accuracy > 0.5 and accuracy < 0.7:
                    buy_shares = 5
                elif accuracy > 0.7 and accuracy < 0.8:
                    buy_shares = 8
                else:
                    buy_shares = 10
                self.Buy_Stock(self.account, self.password, stock_code, buy_shares, stock_information[0]['close'])


    def Get_Stock_Information(self, stock_code, start_date, end_date):
        information_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock_code, start_date, end_date)
        try:
            result = requests.get(information_url).json()
        except:# 處理不在系統的股票
            return dict({})
        if(result['result'] == 'success'):
            return result['data']
        return dict({})

    def Get_User_Stock(self, account, password):
        data = data = {'account':account, 'password':password}
        search_url = "http://140.116.86.242:8081/stock/api/v1/get_user_stocks"
        result = requests.post(search_url, data = data).json()
        if(result['result'] == 'success'):
            return result['data']
        return dict({})

    def Sell_Stock(self, account, password, stock_code, stock_shares, stock_price):
        data = data = {'account':account, 'password':password, 'stock_code':stock_code, 'stock_shares':stock_shares, 'stock_price':stock_price}
        sell_url = 'http://140.116.86.242:8081/stock/api/v1/sell'
        result = requests.post(sell_url, data = data).json()
        if(result['result'] == 'success'):
            print("股票代號: {} 以每張 : {} 賣出: {}張".format(stock_code, stock_price, stock_shares))
        return result['result'] == 'success'

    def Sell(self):
        my_stock = self.Get_User_Stock(self.account, self.password)
        for stock in my_stock:
            stock_information = self.Get_Stock_Information(stock['stock_code_id'], self.start_date, self.end_date)
            if stock['beginning_price'] < stock_information[0]['close']:# 成本價 < 當天收盤價，賣出
                sell_price = stock_information[0]['close']# 以收盤價賣出
                sell_shares = stock['shares']//2 if stock['shares']//2 != 0 else 1# 每次賣出擁有股票的一半
                self.Sell_Stock(self.account, self.password, stock['stock_code_id'], sell_shares, sell_price)

if __name__ == '__main__':
    t = stock_trade()
    t.Buy()
    t.Sell()