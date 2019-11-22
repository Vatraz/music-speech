from keras.callbacks import ModelCheckpoint
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from globals import OUT_PATH, DS_NAME
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model


def main(l):
    # ============================================
    dataset = pd.read_csv(OUT_PATH + l + '.csv', header=0)

    # ============================================
    X = dataset.iloc[:,1:]
    y = dataset.iloc[:,0]

    # ============================================
    corr = dataset.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    # plt.show()

    # ============================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    print(X_train)

    filepath = 'siup.hdf5'
    try:
        model = load_model(filepath)
    except:
        model = Sequential()

        model.add(Dense(5, activation='relu', input_shape=(5,)))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


    checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=False)
    model.fit(X_train, y_train, epochs=3000, batch_size=200, callbacks=[checkpointer])
    #  ===========================================
    score = model.evaluate(X_test, y_test,verbose=1)
    print(score)


def test(model, t):
    #  ===========================================
    dataset = pd.read_csv(OUT_PATH + t + '.csv', header=0)
    X_radio = dataset.iloc[:,1:]
    # print(X_radio)
    y_radio = dataset.iloc[:,0]
    # X_radio = scaler.transform(X_radio)
    score = model.evaluate(X_radio, y_radio,verbose=1)
    if t =='fs':
        hm = (model.predict(X_radio), model.predict_classes(X_radio))
        a = 23

    print(t, ' -> ', score)


def test_all(model, t):
    test(model, t)
    test(model, t+'m')
    test(model, t+'s')

if __name__ == '__main__':
    main('zero')
    model = load_model('siup.hdf5')
    test_all(model, 'wav')
    test_all(model, 'g')
    test_all(model, 'c')
    test_all(model, 't')
    test_all(model, 'aa')
    test_all(model, 'f')