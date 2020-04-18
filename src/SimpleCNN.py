import os
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from utils import *

X_train, y_trainHot = get_data('../data/images/TRAIN/')
X_test, y_testHot = get_data('../data/images/TEST/')

def basicCNN():
    inp = Input(shape=(60,80,3))
    x = Conv2D(16, (7,7), padding="same",activation="relu")(inp)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (5, 5), padding="same",activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding="same",activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(1024,activation="relu")(x)
    y = Dense(5,activation="softmax")(x)
    model = Model(inp, y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model=basicCNN()
model.summary()

filepath = "./weight_tr5.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(X_train,
         y_trainHot,
         epochs = 100,
         batch_size = 1024,
         validation_data = (X_test,y_testHot),
         callbacks = callbacks_list,
         verbose = 1)

print("HISTORY", history.history)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

