import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from globals import OUT_PATH, DS_NAME
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def learn(n):
    df = pd.read_csv(OUT_PATH + n + '.csv', header=0)
    corr = df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    dfm = df.loc[df['type'] == 1]
    dfs = df.loc[df['type'] != 1]
    dfm.hist(bins=50)
    dfs.hist(bins=50)
    plt.show()


learn('zero')

