import os
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, BatchNormalization,\
        Dropout, MaxPooling2D, Input, Softmax, Activation, Flatten
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from utils import *

def cnn_model():
    model = Sequential()
    f, stride = 32, 3
    model.add(Conv2D(f, (stride, stride), activation='relu'))
    m_stride = 2
    model.add(MaxPooling2D((m_stride, m_stride)))

    f, stride = 64, 3
    model.add(Conv2D(f, (stride, stride), activation='relu'))
    m_stride = 2
    model.add(MaxPooling2D((m_stride, m_stride)))

    d = 100
    cat = 5
    model.add(Flatten())
    model.add(Dense(d, activation='relu'))
    model.add(Dense(cat, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def main():
    # get data
    X_train, y_trainHot = get_data('../dataset2-master/images/TRAIN/')
    X_test, y_testHot = get_data('../dataset2-master/images/TEST/')
    # build model
    model = cnn_model()
    history = model.fit(X_train, y_trainHot, epochs=10,
        validation_data=(X_test, y_testHot))
    print(model.summary())

    # train model
    # plot results

if __name__ == "__main__":
    main()
