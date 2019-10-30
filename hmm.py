import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

iris_df = pd.read_csv('iris.txt', names=['sepal_len','sepal_wid','petal_len','petal wid','class'])
pca = PCA(n_components = 2)
pca.fit(iris_df.iloc[:,0:4].values)
X = pca.transform(iris_df.iloc[:,0:4].values)
iris_df['P1'] = X[:,0]
iris_df['P2'] = X[:,1]
iris_df.head(5)
print(iris_df.head(5))

classes = iris_df['class'].unique()

colors = [(0,.8,1),(1,.3,.2),(0,.7, .1)]
plt.figure(figsize = (8,8))
plt.xlabel('P1', fontsize = 15)
plt.ylabel('P2', fontsize = 15)
plt.title('2 component PCA', fontsize = 20)
for cl, color in zip(classes,colors):
    P1 = iris_df[iris_df['class'] == cl]['P1'].values
    P2 = iris_df[iris_df['class'] == cl]['P2'].values
    plt.scatter(P2, P1, c = color, s = 50)
plt.legend(classes)
plt.grid()
plt.show()

df_train = iris_df[iris_df['class'] == classes[0]][0:35]
for c in classes[1:]:
    df_train = pd.concat([df_train, iris_df[iris_df ['class'] == c][0:35]])

df_test = iris_df[iris_df['class'] == classes[0]][35:]
for c in classes[1:]:
    df_test = pd.concat([df_test, iris_df[iris_df['class'] == c][35:]])

def classify(sample_df, valid_df):
    prob = []
    for i in range(3):
        cond = sample_df['class'] == classes[i]
        mean = np.mean(sample_df[cond].ix[:,0:4].values, axis = 0)
        cov = np.cov(np.transpose(sample_df[cond].ix[:,0:4].values))
        func = multivariate_normal(mean =mean, cov=cov)
        hmm = func.logpdf(valid_df.ix[:, 0:4])
        prob.append(func.logpdf(valid_df.ix[:,0:4]))
    max_prob = np.argmax(prob, axis = 0)
    tf_number_error = [classes[i] != j for i,j in zip(max_prob, valid_df['class'])]
    error_percent = np.sum(tf_number_error)/len(valid_df)
    return error_percent

def find_error(sample_df, valid_df):
    prob, label, flower = [], [], []
    for i in range(3):
        cond = sample_df['class'] == classes[i]
        mean = np.mean(sample_df[cond].ix[:,0:4].values, axis = 0)
        cov = np.cov(np.transpose(sample_df[cond].ix[:,0:4].values))
        func = multivariate_normal(mean, cov)

        prob.append(func.logpdf(valid_df.ix[:,0:4]))

    max_prob = np.argmax(prob, axis = 0)

    prob = np.matrix(prob)
    for i,j in zip(max_prob, valid_df['class']):
        if classes[i] != j:
            flower.append(j)
            label.append(classes[i])
    return [flower, label]


[flower, label] = find_error(df_train, df_train)
error_df1 = pd.DataFrame({'prediction': label, 'flower': flower})
error_df1.head()
print(error_df1)
error_percent = classify(df_train, df_test)
print(error_percent)