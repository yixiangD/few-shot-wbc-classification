import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report

from hyper import *
import myplotstyle


def get_data(suffix=''):
    # set up image folder
    train_dir = os.path.join(PATH, 'train'+suffix)
    print(train_dir)
    test_dir = os.path.join(PATH, 'test')
    val_dir = os.path.join(PATH, 'val')

    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
    test_dataset = image_dataset_from_directory(test_dir,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
    val_dataset = image_dataset_from_directory(val_dir,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def transfer_model(image_batch, metrics):
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
    feature_batch = base_model(image_batch)
    print('Feature batch shape: ',feature_batch.shape)
    base_model.trainable = False
    #base_model.summary()

    # add classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

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

    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    #model.summary()
    return base_model, model


def main():
    '''
    Reference:

    https://www.tensorflow.org/tutorials/images/
    transfer_learning#create_the_base_model_from_the_pre-trained_convnets
    '''
    train_dataset, val_dataset, test_dataset = get_data()
    #train_dataset, val_dataset, test_dataset = get_data(suffix='_aug')
    image_batch, label_batch = next(iter(train_dataset))

    #TODO
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

    base_model, model = transfer_model(image_batch, METRICS)

    initial_epochs = 10
    model.evaluate(val_dataset)

    history=model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=val_dataset)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    # Fine-tune from this layer onwards
    fine_tune_at = -1
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

    fine_tune_epochs = 10
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             callbacks = [early_stopping],
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

    model.evaluate(test_dataset)
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()
    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

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
