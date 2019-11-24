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
    plt.show()
    # print(df.describe(include='all'))
    clf = GaussianNB()

    # ===================================
    # One-Hot Encoder - audio type
    # NORMALIZACJA

    # ds = {}
    # for feature in list(df.columns)[1:]:
    #     mean, std = np.mean(df[feature]), np.std(df[feature])
    #     ds[feature] = std
    #     # print(std)
    #     df.loc[:, feature] = (df[feature]) / std
    # ------------
    print(df.describe(include='all'))

    features, target = df.values[:, 1:], df.values[:, 0]
    scaler = None
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)

    # dzielenie na zbior uczacy i testowy
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.29, random_state=100)
    #  ----------
    # print(features)
    # print(target)
    clf.fit(features_train, target_train)

    target_pred = clf.predict(features_test)
    print('acc:', accuracy_score(target_test, target_pred, normalize=True))
    return clf, scaler



# # ==========================
def clas(n, clf, scaler):
    df = pd.read_csv(OUT_PATH + n + '.csv', header=0)

    # NORMALIZACJA
    # for feature in list(df2.columns)[1:]:
    #     # print(feature)
    #     std = ds[feature]
    #     # print(std)
    #     df2.loc[:, feature] = (df2[feature]) / std
    # ------------
    # print(df2.describe(include='all'))
    features, target = df.values[:, 1:], df.values[:, 0]
    if scaler is not None:
        features = scaler.transform(features)

    target_pred = clf.predict(features)
    # target_pred_p = clf.predict_proba(features)
    # for i in zip(target_pred_p, target_pred):
    #     print(i)
    print(n, 'custom acc:', accuracy_score(target, target_pred, normalize=True))


def start_clas(f, clf, scaler):
    clas(f, clf, scaler)
    clas(f + 'm', clf, scaler)
    clas(f + 's', clf, scaler)


clf, scaler = learn('zero')
for f in ('anty', 'jeden', 'dwa', 'trzy', 'olsztyn', 'zet', 'fm', 'classic', 'wav', 'g', 'f', 't', 'c', 'zero'):
    start_clas(f, clf, scaler)

