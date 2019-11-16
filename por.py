import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from globals import OUT_PATH, DS_NAME


def hm(n):
    df = pd.read_csv(OUT_PATH + n, header=0)
    le = preprocessing.LabelEncoder()
    le.fit(df.type)
    type_cat = le.fit_transform(df.type)
    df['type'] = type_cat

    for feature in list(df.columns)[1:]:
        mean = np.mean(df[feature])
        print(feature, mean)


hm('gs.csv')
hm('gm.csv')
