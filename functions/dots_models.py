from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Flatten, Dense, Lambda
import numpy as np
from tensorflow import pad
import tensorflow as tf
from tensorflow import keras
# cnn without padding
def create_small_cnn_np(k,n_hidden=512):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(k, k, 1)))
    model.add(Conv2D(n_hidden, kernel_size=k))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    #model.add(Dense(2))
    model.add(Dense(2, activation="softmax"))
    return model

def create_fcn(k, n_hidden=512):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(k, k, 1)))
    model.add(Flatten())
    model.add(Dense(n_hidden, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model

def circular_padding(x, padding_size):
    # Perform circular padding on the input tensor
    return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='SYMMETRIC')


def create_small_cnn(k, n_hidden=512,kernel_size=17, padding_size=8):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(k, k, 1)))
    model.add(Lambda(lambda x: circular_padding(x, padding_size)))
    model.add(Conv2D(n_hidden, kernel_size=17, padding='SAME')) #VALID
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation="softmax"))

    return model

def create_dataset_dots(k=100):
    input_shape = (k, k)
    X_1 = np.zeros(input_shape)+0.5
    X_2 = X_1.copy()
    X_1[k//2][k//2]+=0.5
    X_2[k//2][k//2]-=0.5
    Xs = np.stack([X_1, X_2], axis=0)
    ys = np.array([1, 0])
    return Xs, ys

def train_dot_model(model, batch_X,batch_y,criterion,opt,LR):
    for idx in range(2000):
        total_loss = 0
        total_acc = 0

        with tf.GradientTape() as tape:
            output = model(batch_X, training=True)
            loss = criterion(batch_y, output)

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss += loss * tf.cast(tf.shape(batch_y)[0], dtype=tf.float32)
        total_acc += tf.reduce_sum(
            tf.cast(tf.math.argmax(output, axis=1) == tf.cast(batch_y, dtype=tf.int64), tf.float32)
        ).numpy()

        if (idx + 1) % 800 == 0:
            LR /= 10
            opt.learning_rate.assign(LR)

        if (idx + 1) % 100 == 0:
            print("Epoch: {}, Train Acc: {:.3f}, Loss: {:.3f}".format(
                idx + 1, total_acc * 100. / batch_y.numpy().shape[0], total_loss / batch_y.numpy().shape[0]
            ))

def train_dot_model_no_verbose(model, batch_X,batch_y,criterion,opt,LR):
    for idx in range(2000):
        with tf.GradientTape() as tape:
            output = model(batch_X, training=True)
            loss = criterion(batch_y, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        if (idx + 1) % 800 == 0:
            LR /= 10
            opt.learning_rate.assign(LR)