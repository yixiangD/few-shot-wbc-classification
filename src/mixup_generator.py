import numpy as np


class MixupGenerator:
    def __init__(
        self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[
                    i * self.batch_size * 2 : (i + 1) * self.batch_size * 2
                ]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = np.copy(l)

        X1 = self.X_train[batch_ids[: self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size :]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[: self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size :]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[: self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size :]]
            y = y1 * y_l + y2 * (1 - y_l)
        return X, y


class MyMixupGenerator:
    """
    This class is adopted from the one shown in
        https://github.com/yu4u/mixup-generator/blob/master/mixup_generator.py
    Differences:
    1) do not support data generator as in the origianl post, simplify for this project
    2) instead of processing two batches at a time, here we only take one batch and shuffle
    the indices
    3) support minority data augmentation for minority class (labled with 0) in binary
    classification
    """

    def __init__(
        self, X_train, y_train, batch_size=32, alpha=0.2, minority=0, shuffle=True
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.minority = minority
        self.sample_num = len(X_train)

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size))
            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size : (i + 1) * self.batch_size]
                if self.minority != 0:
                    X, y = self.__minority_data_generation(batch_ids, self.minority)
                else:
                    X, y = self.__data_generation(batch_ids)
                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, batch_ids):
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = np.copy(l)

        X1 = self.X_train[batch_ids]
        y1 = self.y_train[batch_ids]

        np.random.shuffle(batch_ids)
        X2 = self.X_train[batch_ids]
        y2 = self.y_train[batch_ids]

        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def __minority_data_generation(self, batch_ids, minority):
        X = self.X_train[batch_ids]
        y = self.y_train[batch_ids]
        if 0 in y:
            minor_ids = np.tile(np.where(y == 0)[0], minority)
            l = np.random.beta(self.alpha, self.alpha, len(minor_ids))
            X_l = l.reshape(len(l), 1, 1, 1)

            x1 = X[minor_ids]
            x2 = X[np.random.permutation(minor_ids)]
            X_new = x1 * X_l + x2 * (1 - X_l)
            X = np.concatenate((X, X_new))
            y = np.concatenate((y, np.zeros(len(minor_ids))))
            return X, y
        else:
            return X, y
