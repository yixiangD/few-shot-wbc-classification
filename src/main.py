import os
import argparse
from math import floor, ceil
from collections import defaultdict

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
    data = defaultdict(list)
    cell = ['LYMPHOCYTE', 'nonLYMPHOCYTE']
    for c in range(2):
        img_dir = os.path.join(PATH, cell[c])
        for root, dirs, files in os.walk(img_dir):
            root, dirs, files = root, dirs, files
            break
        for f in files:
            img = np.expand_dims(plt.imread('/'.join([root, f])), axis=0)
            data[c].append(img)
    return data

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
    fp, tp, threshold = sklearn.metrics.roc_curve(labels, predictions)
    #print(fp, tp, threshold)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('FP [%]')
    plt.ylabel('TP [%]')
    plt.xlim([-0.5, 100])
    plt.ylim([0, 100.5])
    plt.legend(loc='lower right')

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
    parser = argparse.ArgumentParser(description='Specify the data folder path')
    parser.add_argument('--method', type=str,
                                    default='none',
                                    choices=['oversample', 'weighted', 'none'],
                                    help='list ways of treating imbalace')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()
    data = get_data()
    for k in data:
        data[k] = np.concatenate(data[k])
    split = 0.7
    n0 = len(data[0])
    n1 = len(data[1])
    if not args.load:
        train_index0 = np.random.choice(np.arange(n0), ceil(split*n0), replace=False)
        train_index1 = np.random.choice(np.arange(n1), ceil(split*n1), replace=False)
        np.savetxt('train_index0.txt', train_index0)
        np.savetxt('train_index1.txt', train_index1)
    else:
        train_index0 = np.loadtxt('train_index0.txt').astype(int)
        train_index1 = np.loadtxt('train_index1.txt').astype(int)
    test_index0 = set(np.arange(n0)).difference(set(train_index0))
    test_index1 = set(np.arange(n1)).difference(set(train_index1))
    print(train_index0)
    train_x0 = data[0][train_index0]
    train_x1 = data[1][train_index1]
    test_x0 = data[0][list(test_index0)]
    test_x1 = data[1][list(test_index1)]
    train_x = np.concatenate((train_x0, train_x1))
    test_x = np.concatenate((test_x0, test_x1))
    train_y = np.concatenate((np.zeros(ceil(split*n0)), np.ones(ceil(split*n1))))
    test_y = np.concatenate((np.zeros(n0 - ceil(split*n0)), np.ones(n1 - ceil(split*n1))))
    n = n0 + n1
    print('#train {}, #test {}, train r {:.4f}, test r {:.4f}'.format(len(train_y),
                                                     len(test_y),
                                                     len(train_y)/n,
                                                     len(test_y)/n))

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
                        validation_split=VAL_SPLIT)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    # Fine tune from this layer onwards
    fine_tune_at = -2
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:fine_tune_at]:
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
    #model.summary()

    total_epochs =  initial_epochs + fine_tune_epochs
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_y),
                                                 train_y)
    class_weights = dict(enumerate(class_weights))
    #class_weights = compute_weight(13, 129)
    if args.method == 'weighted':
        history_fine = model.fit(train_x, train_y,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_split=VAL_SPLIT,
                             class_weight=class_weights,
                             shuffle=True)
    elif args.method == 'oversample':
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
        minority_train_xs = np.copy(minority_train_x)
        for _ in range(n_oversample-1):
            minority_trans = augment(minority_train_x)
            minority_train_xs = np.concatenate((minority_train_xs, minority_trans))
        minority_train_ys = np.tile(minority_train_y, n_oversample)
        train_x = np.concatenate((train_x, minority_train_xs))
        train_y = np.concatenate((train_y, minority_train_ys), axis=None)
        history_fine = model.fit(train_x, train_y,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_split=VAL_SPLIT,
                             shuffle=True)

    else:
        history_fine = model.fit(train_x, train_y,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_split=VAL_SPLIT,
                             shuffle=True)

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

    result = model.evaluate(test_x, test_y)
    predictions = model.predict(test_x)
    plt.figure()
    plot_roc('Test (auc: {:.4f})'.format(result[-1]), test_y, predictions, color='r')
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    print(classification_report(test_y, predictions))

    # reevalute on training data
    result = model.evaluate(train_x, train_y)
    predictions = model.predict(train_x)
    plot_roc('Train (auc: {:.4f})'.format(result[-1]), train_y, predictions, color='k')
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    print(classification_report(train_y, predictions))
    name = PATH.split('/')[-1]
    plt.savefig(f'../figs/{name}_{args.method}.png')
    plt.show()

if __name__ == "__main__":
    main()
