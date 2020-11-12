import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

def img_augmentation(inputs):
    return Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)(inputs)

def build_model(num_classes, lr=1e-4):
    """
    # IMG_SIZE is determined by EfficientNet model choice
    """
    IMG_SIZE = 224
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model

def plot_hist(hist):

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training')
    plt.plot(val_acc, label='Validation')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training')
    plt.plot(val_loss, label='Validation')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.ylim([0, 1.0])
    plt.xlabel('epoch')
    plt.savefig('../figs/initial_train.png')

def run():
    from main import get_data
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    train_dataset, val_dataset, test_dataset = get_data()

    image_batch, label_batch = next(iter(train_dataset))
    model = build_model(2)
    epochs = 10
    hist = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=2)
    loss0, accuracy0 = model.evaluate(val_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))
    plot_hist(hist)

if __name__ == '__main__':
    run()
