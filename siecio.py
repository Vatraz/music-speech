import glob

from tensorflow.python.keras.callbacks import ModelCheckpoint
from numpy import loadtxt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from globals import OUT_PATH, DS_NAME, DATASET_PATH
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import load_model


def main(l, sc = False):
    # ============================================
    dataset = pd.read_csv(OUT_PATH + l + '.csv', header=0)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

    print(X_train)

    filepath = 'hit.hdf5'
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
    history = model.fit(X_train, y_train, verbose=0, epochs=3000, batch_size=400, callbacks=[checkpointer])

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

    # print(t, ' -> ', score)
    return score



def test_all(model, t, sc=None):
    print(t)

    m_f = len(glob.glob(DATASET_PATH + f'music_' + t + '_25_22k/*.wav'))

    a = test(model, t, sc)
    try:
        b = test(model, t+'m', sc)
        c = test(model, t+'s', sc)
        print(f'{a[1]*100:.2f}\\% \\\\')
        print (f'{divmod(m_f*2, 60)[0]}m {divmod(m_f*2, 60)[1]}s & {b[1]*100:.2f}\\% & {divmod(180*60- m_f*2, 60)[0]}m {divmod(180*60- m_f*2, 60)[1]}s & {c[1]*100:.2f}\\% \\\\')
    except:
        pass


if __name__ == '__main__':

    # main('hm', sc=False)

    # model = load_model('pukiconajlepszy550.hdf5')
    # model = load_model('hit.hdf5')
    model = load_model('hit.hdf5')
    for tup in ('hm', 'anty', 'jeden', 'dwa', 'trzy', 'olsztyn', 'zet', 'fm', 'classic','wav', 'g', 't', 'c'):
        test_all(model, tup)
