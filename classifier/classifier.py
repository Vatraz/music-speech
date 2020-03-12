from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

from classifier.globals import MODEL_PATH



class Classifier():
    def __init__(self, model_path=MODEL_PATH):
        self.model = load_model(model_path)

    def predict(self, X):
        return self.model.predict(X)


def train(csv_path):
    try:
        dataset = pd.read_csv(csv_path, header=0)
    except FileNotFoundError:
        print(csv_path + 'not found')
        return

    X = dataset.iloc[:,1:]
    y = dataset.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=997)

    try:
        model = load_model(MODEL_PATH)
    except:
        model = Sequential()

        model.add(Dense(5, activation='relu', input_shape=(5,)))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    checkpointer = ModelCheckpoint(MODEL_PATH, verbose=1, save_best_only=True, monitor='acc', save_weights_only=False)
    model.fit(X_train, y_train, epochs=500, batch_size=200, callbacks=[checkpointer])

    score = model.evaluate(X_test, y_test,verbose=1)
    print(score)


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 2 and sys.argv[1].endswith('csv'):
        train(sys.argv[1])
    else:
        print('Enter the path of a .csv file')
