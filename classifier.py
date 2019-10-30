import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv', header=0)
# print(df.describe(include='all'))

# One-Hot Encoder - audio type
le = preprocessing.LabelEncoder()
le.fit(df.type)
type_cat = le.fit_transform(df.type)
df['type'] = type_cat

for feature in list(df.columns)[1:]:
    mean, std = np.mean(df[feature]), np.std(df[feature])
    # df.loc[:, feature] = (df[feature]) / std

print(df)
features, target = df.values[:, 1:], df.values[:, 0]


features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.29, random_state=6625)
clf = GaussianNB()
clf.fit(features_train, target_train)

target_pred = clf.predict(features_test)
print('acc:', accuracy_score(target_test, target_pred, normalize=True))
# # ==========================
df2 = pd.read_csv('dataset_musan.csv', header=0)
le = preprocessing.LabelEncoder()
le.fit(df2.type)
type_cat = le.fit_transform(df2.type)
df2['type'] = type_cat
for feature in list(df2.columns)[1:]:
    mean, std = np.mean(df[feature]), np.std(df[feature])
    # df2.loc[:, feature] = (df2[feature]) / std
features2, target2 = df2.values[:, 1:], df2.values[:, 0]

target_pred = clf.predict(features2)
print('custom acc:', accuracy_score(target2, target_pred, normalize=True))
# # ==========================

