from technical_indicators.BBI import  Get_Stock_BBI
from technical_indicators.BIAS import  Get_Stock_BIAS
from technical_indicators.CDP import  Get_Stock_CDP
from technical_indicators.DMI import  Get_Stock_DMI
from technical_indicators.KDJ import  Get_Stock_KDJ
from technical_indicators.MA import  Get_Stock_MA
from technical_indicators.MACD import  Get_Stock_MACD
from technical_indicators.RSI import  Get_Stock_RSI
from technical_indicators.William import  Get_Stock_William
import pandas as pd


# start_date 開始日期，YYYYMMDD
# stop_date  結束日期，YYYYMMDD
# stock_code 股票ID 
def Get_Stock_Technical_Indicators(stock_code, start_date, stop_date):
    BIAS = pd.read_json(Get_Stock_BIAS(stock_code, start_date, stop_date))
    CDP = pd.read_json(Get_Stock_CDP(stock_code, start_date, stop_date))
    DMI = pd.read_json(Get_Stock_DMI(stock_code, start_date, stop_date))
    KDJ = pd.read_json(Get_Stock_KDJ(stock_code, start_date, stop_date))
    MA = pd.read_json(Get_Stock_MA(stock_code, start_date, stop_date))
    MACD = pd.read_json(Get_Stock_MACD(stock_code, start_date, stop_date))
    RSI = pd.read_json(Get_Stock_RSI(stock_code, start_date, stop_date))
    William = pd.read_json(Get_Stock_William(stock_code, start_date, stop_date))
    df = pd.DataFrame()
    #date
    df['date'] = BIAS['date']

    #BIAS
    df['BIAS10'] = BIAS['BIAS10']
    df['BIAS20'] = BIAS['BIAS20']
    df['B10-B20'] = BIAS['B10-B20']

    #CDP
    df['AH'] = CDP['AH']
    df['NH'] = CDP['NH']
    df['NL'] = CDP['NL']
    df['AL'] = CDP['AL']

    #DMI
    df['+DI'] = DMI['+DI']
    df['-DI'] = DMI['-DI']
    df['ADX'] = DMI['ADX']

    #KDJ
    df['RSV'] = KDJ['RSV']
    df['K9'] = KDJ['K9']
    df['D9'] = KDJ['D9']
    df['J9'] = KDJ['J9']
    df['3K-2D'] = KDJ['3K-2D']

    #MA
    df['MA5'] = MA['MA5']
    df['MA20'] = MA['MA20']
    df['MA60'] = MA['MA60']

    #MACD
    df['EMA12'] = MACD['EMA12']
    df['EMA26'] = MACD['EMA26']
    df['DIF9'] = MACD['DIF9']
    df['MACD'] = MACD['MACD']
    df['OSC'] = MACD['OSC']

    #RSI
    df['RSI5'] = RSI['RSI5']

    #William
    df['W%R9'] = William['W%R9']
    return df


if __name__ == '__main__':
    df = Get_Stock_Technical_Indicators('2330', '20220101', '20220613')
    print(df)