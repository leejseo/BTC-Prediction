import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

#Crawl value of BTC in KRW

def _norm(x, min_val, max_val):
    denom = max_val - min_val
    sub = min_val
    return (x - sub)/denom

def crawl_btc_to_df():
    base_url = 'https://min-api.cryptocompare.com/data/histoday'
    result = requests.get(base_url + '?fsym=BTC&tsym=KRW&limit=2000')
    df = pd.DataFrame(json.loads(result.content)['Data'])
    df = df.loc[:, ['time','close']]
    #df = df.astype('float')
    #min_val = df['time'].min()
    #max_val = df['time'].max()
    #norm = lambda x : _norm(x, min_val, max_val)
    #df['time'] = df['time'].map(norm)
    df = df.set_index('time')
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