import numpy as np
from globals import OUT_PATH, DS_NAME
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(OUT_PATH + 'g.csv', header=0)
ds = {}
for feature in list(df.columns)[1:]:
    mean, std = np.mean(df[feature]), np.std(df[feature])
    ds[feature] = std
    # print(std)
    df.loc[:, feature] = (df[feature]) / std



df2 = pd.read_csv(OUT_PATH + 'gm.csv', header=0)
for feature in list(df2.columns)[1:]:
    df2.loc[:, feature] = (df2[feature]) / ds[feature]
df2.hist(bins=100)

df3 = pd.read_csv(OUT_PATH + 'gs.csv', header=0)
for feature in list(df3.columns)[1:]:
    df3.loc[:, feature] = (df3[feature]) / ds[feature]
df3.hist(bins=100)
plt.show()












df = pd.read_csv(OUT_PATH + 'g.csv', header=0)
ds = {}
for feature in list(df.columns)[1:]:
    mean, std = np.mean(df[feature]), np.std(df[feature])
    ds[feature] = std
    # print(std)
    df.loc[:, feature] = (df[feature]) / std



df2 = pd.read_csv(OUT_PATH + 'gm.csv', header=0)
for feature in list(df2.columns)[1:]:
    df2.loc[:, feature] = (df2[feature]) / ds[feature]
df2.hist(bins=100)

df3 = pd.read_csv(OUT_PATH + 'gs.csv', header=0)
for feature in list(df3.columns)[1:]:
    df3.loc[:, feature] = (df3[feature]) / ds[feature]
df3.hist(bins=100)
plt.show()
