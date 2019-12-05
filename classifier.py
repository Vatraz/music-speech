from tensorflow.python.keras.callbacks import ModelCheckpoint
from numpy import loadtxt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from globals import OUT_PATH, DS_NAME
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import load_model


class Classifier():
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, X):
        return self.model.predict(X)



def train(l=None, sc = False):
    dataset = pd.read_csv(OUT_PATH + 'hm' + '.csv', header=0)

    X = dataset.iloc[:,1:]
    y = dataset.iloc[:,0]

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

if __name__ == '__main__':
    train()
    # main('zero', sc=False)
    # #
    # model = load_model('siup.hdf5')
    # for tup in ('anty', 'jeden', 'dwa', 'trzy', 'olsztyn', 'zet', 'fm', 'classic'):
    #     test_all(model, tup)
