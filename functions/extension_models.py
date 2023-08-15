import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, GlobalAveragePooling2D, Dense, Lambda, Flatten, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentTensorFlowV2
from tqdm import tqdm
import os
from art.utils import load_mnist
from tests.utils import master_seed
from art.estimators.classification import TensorFlowV2Classifier
import pickle

def simple_FC(n_hidden):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(n_hidden, activation="relu"))
    model.add(Dense(10))
    return model



def circular_padding(x, padding_size):
    # Perform circular padding on the input tensor
    return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='SYMMETRIC')


def simple_Conv(n_hidden, kernel_size=10, padding_size=-1,n_layers =1,max_pooling= True, add_dense = False ):
    if padding_size == -1:
        padding_size = kernel_size // 2

    model = Sequential()
    model.add(Lambda(lambda x: circular_padding(x, padding_size), input_shape=(28, 28, 1)))
    if n_layers==1:
        model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='same'))
        model.add(ReLU())
        if max_pooling:
            model.add(GlobalMaxPooling2D())
        else:
            model.add(GlobalAveragePooling2D())
    elif n_layers>1:
        for _ in range(n_layers):
            model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='same'))
            model.add(ReLU())
            if max_pooling:
                model.add(MaxPooling2D((2, 2))) # (3,3)
            else:
                model.add(AveragePooling2D((2, 2)))
        model.add(Flatten())
    if add_dense:
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
    model.add(Dense(10))

    return model