import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from globals import OUT_PATH, DS_NAME
from sklearn.preprocessing import StandardScaler

def learn(n):
    df = pd.read_csv(OUT_PATH + n + '.csv', header=0)
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

    features, target = df.values[:, -2:], df.values[:, 0]
    scaler = None
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)

    # dzielenie na zbior uczacy i testowy
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.29, random_state=100)
    #  ----------
    print(features)
    print(target)
    clf.fit(features_train, target_train)

    target_pred = clf.predict(features_test)
    print('acc:', accuracy_score(target_test, target_pred, normalize=True))
    return clf, scaler



# # ==========================
def clas(n, clf, scaler):
    df2 = pd.read_csv(OUT_PATH + n + '.csv', header=0)

    # NORMALIZACJA
    # for feature in list(df2.columns)[1:]:
    #     # print(feature)
    #     std = ds[feature]
    #     # print(std)
    #     df2.loc[:, feature] = (df2[feature]) / std
    # ------------
    # print(df2.describe(include='all'))
    features2, target2 = df2.values[:, -2:], df2.values[:, 0]
    if scaler is not None:
        features2 = scaler.transform(features2)


    target_pred2 = clf.predict(features2)
    # target_pred2_p = clf.predict_proba(features2)
    # print(target_pred2)
    # for i in zip(target_pred2_p, target_pred2):
    #     if i[1] == 0.0:
    #         print(i, '<================================')
    #     else:
    #         print(i)
    print(n, 'custom acc:', accuracy_score(target2, target_pred2, normalize=True))


clf, sclaer = learn('zero')

f = 'g'
clas(f, clf, sclaer)
clas(f+'m', clf, sclaer)
clas(f+'s', clf, sclaer)

f = 't'
clas(f, clf, sclaer)
clas(f+'m', clf, sclaer)
clas(f+'s', clf, sclaer)

f = 'r'
clas(f, clf, sclaer)
clas(f+'m', clf, sclaer)
clas(f+'s', clf, sclaer)