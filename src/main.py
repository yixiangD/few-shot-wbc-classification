import argparse
import os
from collections import Counter, defaultdict
from math import ceil, floor

import cv2
import matplotlib.pyplot as plt
import myplotstyle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from hyper import (
    BATCH_SIZE,
    IMG_SHAPE,
    IMG_SIZE,
    LEARNING_RATE,
    PATH1,
    PATH2,
    VAL_SPLIT,
    fine_tune_epochs,
    initial_epochs,
)
from mixup_generator import MixupGenerator, MyMixupGenerator
from numpy.random import MT19937, RandomState, SeedSequence
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.utils import augment, plot_roc


def get_data(path):
    data = defaultdict(list)
    for root, dirs, files in os.walk(path):
        classes = dirs
        break
    for cls in classes:
        img_path = os.path.join(path, cls)
        for root, dirs, files in os.walk(img_path):
            root, dirs, files = root, dirs, files
            break
        for f in files:
            img = np.expand_dims(plt.imread(os.path.join(root, f)), axis=0)
            data[cls].append(img)
    return data


def transfer_model(metrics):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # create a base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    AUGMENT = False
    if AUGMENT:
        x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    dim = 128
    for _ in range(3):
        x = tf.keras.layers.Dense(dim)(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return base_model, model


def main():
    """
    Reference:

    https://www.tensorflow.org/tutorials/images/
    transfer_learning#create_the_base_model_from_the_pre-trained_convnets
    """
    split = 0.7
    RandomState(MT19937(SeedSequence(0)))
    parser = argparse.ArgumentParser(description="Specify the data folder path")
    parser.add_argument(
        "--method",
        type=str,
        default="none",
        choices=["oversample", "weighted", "none", "mixup"],
        help="list ways of treating imbalace",
    )
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--dataset", type=str, default="raw", choices=["raw", "mask"])
    parser.add_argument("--alpha", type=float, default=0.2)
    args = parser.parse_args()
    if args.dataset == "raw":
        path = PATH1
    else:
        path = PATH2
    data = get_data(path)
    train_x, test_x = [], []
    train_y_str, test_y_str = [], []
    train_index = dict()
    test_index = dict()
    for k in list(data.keys()):
        # for x in data[k]:
        #    print(x.shape)
        imgs = [
            cv2.resize(np.squeeze(x), IMG_SIZE, interpolation=cv2.INTER_AREA)
            for x in data[k]
        ]
        # reconcatenate the img such that it looks like N x W x H x C
        # N: number of images, W: width, H: height, C: channel
        index = np.arange(len(imgs))
        np.random.shuffle(index)
        train_index[k] = index[: int(split * len(imgs))]
        test_index[k] = index[int(split * len(imgs)) :]
        train_x.append(data[k][train_index[k], :, :, :].astype("float32") / 255)
        test_x.append(data[k][test_index[k], :, :, :].astype("float32") / 255)
        train_y_str += [k] * len(train_index[k])
        test_y_str += [k] * len(test_index[k])
    train_x, test_x = np.vstack(train_x), np.vstack(test_x)
    # obtain train_x, train_y, test_x, test_y
    n = len(train_y_str) + len(test_y_str)
    print("#" * 50)
    print(" " * 50)
    print(
        "#train {}, #test {}, train r {:.4f}, test r {:.4f}".format(
            len(train_y_str), len(test_y_str), len(train_y_str) / n, len(test_y_str) / n
        )
    )
    print("#train {}, #test {}, in X view".format(len(train_x), len(test_x)))

    # encoding str to labels
    le = LabelEncoder()
    le.fit(train_y_str)
    train_y = le.transform(train_y_str)
    test_y = le.transform(test_y_str)
    print("After label encoding, train ", Counter(train_y))
    print("After label encoding, test ", Counter(test_y))
    print(" " * 50)
    print("#" * 50)

    METRICS = [
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    base_model, model = transfer_model(METRICS)

    history = model.fit(
        train_x, train_y, epochs=initial_epochs, validation_split=VAL_SPLIT
    )
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    # Fine tune from this layer onwards
    fine_tune_at = -2
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #   monitor="val_auc", verbose=1, patience=10, mode="max", restore_best_weights=True
    # )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(lr=LEARNING_RATE / 10),
        metrics=METRICS,
    )
    # model.summary()

    total_epochs = initial_epochs + fine_tune_epochs
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(train_y), y=train_y
    )
    class_weights = dict(enumerate(class_weights))
    # class_weights = compute_weight(13, 129)
    if args.method == "weighted":
        history_fine = model.fit(
            train_x,
            train_y,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            validation_split=VAL_SPLIT,
            class_weight=class_weights,
            shuffle=True,
        )
    elif args.method == "oversample":
        """
        train_ds = tfds.as_numpy(train_dataset)
        train_x, train_y = [], []
        for ex in train_ds:
            train_x.append(ex[0])
            train_y.append(ex[1])
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y, axis=None)
        """
        minority_train_idx = np.where(train_y == 0)
        minority_train_x = train_x[minority_train_idx]
        minority_train_y = train_y[minority_train_idx]
        n_oversample = 4
        minority_train_xs = np.copy(minority_train_x)
        for _ in range(n_oversample - 1):
            minority_trans = augment(minority_train_x)
            minority_train_xs = np.concatenate((minority_train_xs, minority_trans))
        minority_train_ys = np.tile(minority_train_y, n_oversample)
        train_x = np.concatenate((train_x, minority_train_xs))
        train_y = np.concatenate((train_y, minority_train_ys), axis=None)
        history_fine = model.fit(
            train_x,
            train_y,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            validation_split=VAL_SPLIT,
            shuffle=True,
        )
    elif args.method == "mixup":
        nmixup = 0
        train_gen = MyMixupGenerator(
            train_x, train_y, batch_size=BATCH_SIZE, alpha=args.alpha, minority=nmixup
        )()
        history_fine = model.fit(
            train_gen,
            steps_per_epoch=10,
            initial_epoch=history.epoch[-1],
            epochs=total_epochs,
            validation_data=(test_x, test_y),
            shuffle=True,
        )
    else:
        history_fine = model.fit(
            train_x,
            train_y,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            validation_split=VAL_SPLIT,
            shuffle=True,
        )

    acc += history_fine.history["accuracy"]
    val_acc += history_fine.history["val_accuracy"]

    loss += history_fine.history["loss"]
    val_loss += history_fine.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training")
    plt.plot(val_acc, label="Validation")
    plt.xticks(np.arange(0, initial_epochs + fine_tune_epochs + 1, 5))
    plt.ylim([0.8, 1])
    plt.ylabel("Accuracy")
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
    )
    plt.legend(loc="lower right")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training")
    plt.plot(val_loss, label="Validation")
    plt.ylim([0, 1.0])
    plt.xticks(np.arange(0, initial_epochs + fine_tune_epochs + 1, 5))
    plt.ylabel("Loss")
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
    )
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    # plt.savefig('./figs/full_train.png')

    print(test_x.shape, test_y.shape)
    result = model.evaluate(test_x, test_y)
    predictions = model.predict(test_x)
    plt.figure()
    plot_roc("Test (auc: {:.4f})".format(result[-1]), test_y, predictions, color="r")
    # predictions = tf.nn.sigmoid(predictions)
    # predictions = tf.where(predictions < 0.5, 0, 1)
    # print(classification_report(test_y, predictions))

    # reevalute on training data
    train_result = model.evaluate(train_x, train_y)
    predictions = model.predict(train_x)
    plot_roc(
        "Train (auc: {:.4f})".format(train_result[-1]), train_y, predictions, color="k"
    )
    tab = f"results/{args.dataset}_{args.method}.txt"
    if args.method == "mixup":
        tab = f"results/{args.dataset}_{args.method}_{args.alpha}.txt"
    with open(tab, "a") as infile:
        np.savetxt(infile, (result, train_result), fmt="%.4f")
    # predictions = tf.nn.sigmoid(predictions)
    # predictions = tf.where(predictions < 0.5, 0, 1)
    # print(classification_report(train_y, predictions))
    if args.method == "mixup":
        plt.savefig(f"./figs/{args.dataset}_{args.method}{nmixup}{args.alpha}.png")
    else:
        plt.savefig(f"./figs/{args.dataset}_{args.method}.png")
    # plt.show()


if __name__ == "__main__":
    main()
