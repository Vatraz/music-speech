from functions import get_audio_features
from globals import DATASET_PATH, FREQUENCY, FRAME_WIDTH
import numpy as np
from numpy import loadtxt
from globals import OUT_PATH, DS_NAME
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from tensorflow.python.keras import models
import matplotlib.dates as mdates
import datetime

myFmt = mdates.DateFormatter('%H:%M')


df = pd.read_csv('kasz.csv')
df.columns = ['t', 'y']
df['y'] = df['y'].map(lambda x: float(x[2:-2]))

tt = df['t'].values


for i in range(len(tt)-1):
    tt[i] = tt[i+1] - tt[i]
print(tt)
print(np.mean(tt[:-1]))

# t = [datetime.datetime.fromtimestamp(tt).strftime('%H:%M:%S') for tt in df['t'].values]
# print(df)
#
# plt.plot(t[:100], df['y'].values[:100])
# plt.xticks(np.arange(0, 100, 10))
# plt.gcf().autofmt_xdate()
# plt.show()
