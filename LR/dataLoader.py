#!/usr/bin/python3
import pandas as pd

class data_loader:
    def __init__(self, test_split):
        self.x_train = None
        self.y_train = None
        
        self.x_test = None
        self.y_test = None

        assert 0 <= test_split and test_split < 1
        self.split = test_split

    def graduation(self):
        df = pd.read_csv('data/graduation/admissions.csv')
        y = df['Chance of Admit ']
        x = df[df.columns.difference(['Change of Admit '])]
        
        s = int(df.shape[0]*self.split)
        te = [int(i) for i in range(s)]
        tr = [int(i) for i in range(s, df.shape[0])]
        
        self.x_train, self.y_train = x.iloc[te], y.iloc[te]
        self.x_test, self.y_test = x.iloc[tr], y.iloc[tr]
        assert self.x_train.shape[0] + self.x_test.shape[0] == x.shape[0]
        assert self.y_train.shape[0] + self.y_test.shape[0] == y.shape[0]


    def load_data(self):
        return (self.x_train.values, self.y_train.values), (self.x_test.values,
                self.y_test.values)
