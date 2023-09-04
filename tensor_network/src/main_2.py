import os
import random
from dataclasses import dataclass
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# from keras.callbacks import ModelCheckpoint
from keras.datasets import fashion_mnist
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from numpy.typing import NDArray

import src.tensor_network_layer_for_mnist as tnl

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
NUM_CLASSES: Final = 10


def set_seed(seed: int = 200) -> None:
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_normal_network() -> tf.keras.Sequential:
    model = Sequential()  # Declare a Sequential model
    model.add(Dense(512, input_shape=(784,)))  # Add a dense input layer of 512 units
    # The input shape is (784,) because the features are 784 dimensional vectors
    model.add(Activation("relu"))  # Apply ReLu activation function to the layer output
    model.add(Dropout(0.2))  # Dropout helps prevent the model from overfitting on the training data
    model.add(Dense(512))  # Add a hidden layer of 512 units
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES))  # Add an output layer of 10 units, the number of classes
    model.add(
        Activation("softmax")
    )  # The "softmax" activation ensures the output is a valid probability distribution

    return model


def make_network_with_tensor_network() -> tf.keras.Sequential:
    model = Sequential()  # Declare a Sequential model
    model.add(tnl.TNLayer())
    model.add(Activation("relu"))  # Apply ReLu activation function to the layer output
    model.add(Dropout(0.2))  # Dropout helps prevent the model from overfitting on the training data
    model.add(Dense(512))  # Add a hidden layer of 512 units
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES))  # Add an output layer of 10 units, the number of classes
    model.add(
        Activation("softmax")
    )  # The "softmax" activation ensures the output is a valid probability distribution
    return model


@dataclass
class Dataset:
    X_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.float64]


def load_dataset() -> Dataset:
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return Dataset(X_train, y_train, X_test, y_test)


def modify_dataset_format(dataset: Dataset) -> Dataset:
    X_train = dataset.X_train.reshape(dataset.X_train.shape[0], 784)
    X_test = dataset.X_test.reshape(dataset.X_test.shape[0], 784)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    num_classes = 10  # There are 10 classes possible
    y_train = np_utils.to_categorical(dataset.y_train, num_classes)
    y_test = np_utils.to_categorical(dataset.y_test, num_classes)
    return Dataset(X_train, y_train, X_test, y_test)


def train(model: tf.keras.Sequential, epochs: int, dataset: Dataset) -> Any:
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    return model.fit(
        dataset.X_train,
        dataset.y_train,
        batch_size=128,
        epochs=epochs,
        verbose=1,
        validation_data=(dataset.X_test, dataset.y_test),
    )


def plot_history(history: Any, name: str) -> None:
    plt.plot(range(1, 1 + len(history.history["accuracy"])), history.history["accuracy"])
    plt.plot(range(1, 1 + len(history.history["val_accuracy"])), history.history["val_accuracy"])
    plt.title("Model Accuracy vs Number of Epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch Number")
    plt.legend(["train", "test"], loc="best")
    plt.xticks(np.arange(0, 28, 2))
    plt.savefig(f"./images/accuracy_{name}.png")


if __name__ == "__main__":
    set_seed()
    print("_/_/_/ make dataset")
    dataset = load_dataset()
    modified_dataset = modify_dataset_format(dataset)

    NAME = "tensor_network"
    MAKE_NETWORK = {
        "normal": make_normal_network,
        "tensor_network": make_network_with_tensor_network,
    }

    print("_/_/_/ make model")
    model = MAKE_NETWORK[NAME]()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("_/_/_/ train model")
    epochs = 1
    history = train(model, epochs, modified_dataset)
    model.summary()

    # print("_/_/_/ plot history")
    # plot_history(history, NAME)
