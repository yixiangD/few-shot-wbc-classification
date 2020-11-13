import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_datasets as tfds
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

from hyper import *
import myplotstyle


def get_data(suffix=''):
    # set up image folder
    X, y = [], []
    cell = ['LYMPHOCYTE', 'nonLYMPHOCYTE']
    for c in range(2):
        img_dir = os.path.join(PATH, cell[c])
        for root, dirs, files in os.walk(img_dir):
            root, dirs, files = root, dirs, files
            break
        for f in files:
            img = np.expand_dims(plt.imread('/'.join([root, f])), axis=0)
            X.append(img)
            y.append(c)
    return X, y

def transfer_model(metrics):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

    # create a base model
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    #x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = rescale(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    dim = x.shape[-1]
    for _ in range(3):
        x = tf.keras.layers.Dense(dim)(x)

    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    return base_model, model

def compute_weight(pos, neg):
    total = pos + neg
    weight0 = (1/neg)*total/2.0
    weight1 = (1/pos)*total/2.0
    class_weight = {0 : weight0, 1 : weight1}
    return class_weight

def plot_roc(name, labels, predictions, **kwargs):
    plt.figure()
    fp, tp, threshold = sklearn.metrics.roc_curve(labels, predictions)
    print(fp, tp, threshold)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 100])
    plt.ylim([0, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def augment(imgs):
    res = []
    for i in range(imgs.shape[0]):
        image = imgs[i]
        image_shape = image.shape
        image = tf.image.resize_with_crop_or_pad(image, image_shape[0] + 6, image_shape[1] + 6)
        # Random crop back to the original size
        image = tf.image.random_crop(image, size=image_shape)
        image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
        res.append(np.expand_dims(image, axis=0))
    return np.concatenate(res)

def main():
    '''
    Reference:

    https://www.tensorflow.org/tutorials/images/
    transfer_learning#create_the_base_model_from_the_pre-trained_convnets
    '''
    X, y = get_data()
    X = np.concatenate(X)
    y = np.concatenate(y, axis=None)
    n_img = len(y)
    train_y, test_y = [], []
    while 0 not in train_y or 0 not in test_y:
        train_index = np.random.choice(np.arange(n_img), int(0.7*n_img))
        test_index = [x for x in np.arange(n_img) if x not in train_index]
        train_y = y[train_index]
        test_y = y[test_index]
    train_x = X[train_index]
    test_x = X[test_index]

    METRICS = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            ]

    base_model, model = transfer_model(METRICS)

    history=model.fit(train_x, train_y,
                        epochs=initial_epochs,
                        validation_split=0.1)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    # Fine tune from this layer onwards
    fine_tune_at = -2
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer = tf.keras.optimizers.RMSprop(lr=LEARNING_RATE/10),
                  metrics=METRICS)
    model.summary()

    total_epochs =  initial_epochs + fine_tune_epochs
    weighted = True
    oversample = True
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_y),
                                                 train_y)
    class_weights = dict(enumerate(class_weights))
    #class_weights = compute_weight(13, 129)
    if weighted:
        history_fine = model.fit(train_x, train_y,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_split=0.2,
                             class_weight=class_weights)
    elif oversample:
        '''
        train_ds = tfds.as_numpy(train_dataset)
        train_x, train_y = [], []
        for ex in train_ds:
            train_x.append(ex[0])
            train_y.append(ex[1])
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y, axis=None)
        '''
        minority_train_idx = np.where(train_y == 0)
        minority_train_x = train_x[minority_train_idx]
        minority_train_y = train_y[minority_train_idx]
        n_oversample = 4
        print(train_x.shape, train_y.shape)
        print(minority_train_x.shape)
        minority_train_xs = np.copy(minority_train_x)
        for _ in range(n_oversample):
            minority_trans = augment(minority_train_x)
            minority_train_xs = np.concatenate((minority_train_xs, minority_trans))
        minority_train_ys = np.tile(minority_train_y, n_oversample + 1)
        train_x = np.concatenate((train_x, minority_train_xs))
        train_y = np.concatenate((train_y, minority_train_ys), axis=None)
        print(train_x.shape, train_y.shape)
        history_fine = model.fit(train_x, train_y,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_split=0.1)
 
    else:
        history_fine = model.fit(train_x, train_y,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=val_dataset)

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training')
    plt.plot(val_acc, label='Validation')
    plt.xticks(np.arange(0, initial_epochs + fine_tune_epochs + 1, 5))
    plt.ylim([0.8, 1])
    plt.ylabel('Accuracy')
    plt.plot([initial_epochs-1,initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training')
    plt.plot(val_loss, label='Validation')
    plt.ylim([0, 1.0])
    plt.xticks(np.arange(0, initial_epochs + fine_tune_epochs + 1, 5))
    plt.ylabel('Loss')
    plt.plot([initial_epochs-1,initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.savefig('../figs/full_train.png')

    model.evaluate(test_x, test_y)
    predictions = model.predict(test_x)
    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    plot_roc('test', test_y, predictions, color='k')
    plt.show()
    exit()

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)
    print(classification_report(label_batch, predictions.numpy()))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
