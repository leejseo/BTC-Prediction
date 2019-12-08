import requests
import json
import pandas as pd

#Crawl value of BTC in KRW

def crawl_btc_to_df():
    base_url = 'https://min-api.cryptocompare.com/data/histoday'
    result = requests.get(base_url + '?fsym=BTC&tsym=KRW&limit=2000')
    df = pd.DataFrame(json.loads(result.content)['Data'])
    df = df.loc[:, ['time','close']]
    df = df.set_index('time')
    df.index = pd.to_datetime(df.index, unit = 's')
    return df

def train_test_split(df, r=0.2):
    numTrain = len(df) - max(1, int(r*len(df)))
    train = df.iloc[:numTrain]
    test = df.iloc[numTrain:]
    return train, test

def save_train_test(train, test):
    train.to_csv("train.csv", mode="w")
    test.to_csv("test.csv", mode="w")

if __name__ == '__main__':
    df = crawl_btc_to_df()
    train, test = train_test_split(df)
    save_train_test(train, test)