from keras.callbacks import ModelCheckpoint
from numpy import loadtxt
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
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


def main(l, sc = False):
    # ============================================
    dataset = pd.read_csv(OUT_PATH + 'hm' + '.csv', header=0)

    # ============================================
    X = dataset.iloc[:,1:]
    y = dataset.iloc[:,0]

    # ============================================
    # corr = dataset.corr()
    # sns.heatmap(corr,
    #             xticklabels=corr.columns.values,
    #             yticklabels=corr.columns.values)
    # plt.show()

    # ============================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=997)

    print(X_train)

    filepath = 'siup.hdf5'
    try:
        model = load_model(filepath)
    except:
        model = Sequential()

        model.add(Dense(5, activation='relu', input_shape=(5,)))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


    checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True, monitor='acc', save_weights_only=False)
    history = model.fit(X_train, y_train, epochs=500, batch_size=200, callbacks=[checkpointer])

    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    #  ===========================================
    score = model.evaluate(X_test, y_test,verbose=1)
    print(score)



def test(model, t, scaler = None):
    #  ===========================================
    dataset = pd.read_csv(OUT_PATH + t + '.csv', header=0)
    X_radio = dataset.iloc[:,1:]
    # print(X_radio)
    y_radio = dataset.iloc[:,0]

    score = model.evaluate(X_radio, y_radio,verbose=0)

    print(t, ' -> ', score)


def test_sieci():
    dataset = pd.read_csv(OUT_PATH + 'zero' + '.csv', header=0)

    # ============================================
    X = dataset.iloc[:,2:]
    y = dataset.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=997)

    print(X_train)
    for i in range(19, 20, 1):
        model = Sequential()

        model.add(Dense(5, activation='relu', input_shape=(5,)))
        model.add(Dense(i, activation='relu'))
        model.add(Dense(i, activation='relu'))
        model.add(Dense(i, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1000, batch_size=200, verbose=0)

        score = model.evaluate(X_test, y_test, verbose=0)
        print('1', i, score)


    for i in range(12, 25, 1):
        model = Sequential()

        model.add(Dense(5, activation='relu', input_shape=(5,)))
        model.add(Dense(2*i, activation='relu'))
        model.add(Dense(2*i, activation='relu'))
        model.add(Dense(i, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1000, batch_size=200, verbose=0)

        score = model.evaluate(X_test, y_test, verbose=0)
        print('3', i, score)
    # checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True, monitor='acc', save_weights_only=False)


def test_all(model, t, sc=None):
    test(model, t, sc)
    test(model, t+'m', sc)
    test(model, t+'s', sc)

if __name__ == '__main__':
    # test_sieci()
    # main('hm', sc=False)
    # #
    model = load_model('siup.hdf5')
    for tup in ('anty', 'jeden', 'dwa', 'trzy', 'olsztyn', 'zet', 'fm', 'classic'):
        test_all(model, tup)
