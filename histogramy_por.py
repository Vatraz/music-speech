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


learn('zero')

