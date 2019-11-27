import pandas as pd
from globals import OUT_PATH, DS_NAME
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

def por(n):

    df = pd.read_csv(OUT_PATH + n + '.csv', header=0)
    corr = df.corr()
    # sns.heatmap(corr,
    #             xticklabels=corr.columns.values,
    #             yticklabels=corr.columns.values)
    dfm = df.loc[df['type'] == 1]
    dfs = df.loc[df['type'] != 1]
    dfm = dfm['zcr_std_of_fod']
    dfm = dfm[:2700]
    dfs = dfs['zcr_std_of_fod']
    dfs = dfs[:2700]
    dfm.hist(bins=40, alpha=0.7, color='b', label='utwory muzyczne', edgecolor='b')
    dfs.hist(bins=40, alpha=0.7, color='r', label='treści informacyjne', edgecolor='r')
    plt.ylabel('liczność')
    plt.xlabel('wartość')
    plt.xlim([0, 0.08])
    plt.legend()
    plt.show()

learn('hm')
por('hm')

