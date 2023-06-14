import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, GlobalAveragePooling2D, Dense, Lambda, Flatten
from tqdm.contrib import tzip
import os
import pickle
from art.utils import load_mnist
from tensorflow.keras.callbacks import TensorBoard


print(" Set GPU devices")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


print("Loading MNIST dataset")
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

print("Defining models")
def circular_padding(x, padding_size):
    # Perform circular padding on the input tensor
    return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='SYMMETRIC')

def simple_Conv(n_hidden, kernel_size=28, padding_size=-1):
    if padding_size == -1:
        padding_size = kernel_size // 2

    model = Sequential()
    model.add(Lambda(lambda x: circular_padding(x, padding_size), input_shape=(28, 28, 1)))
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='valid'))
    model.add(ReLU())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10))

    return model

def simple_FC(n_hidden, n_units = 2):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for _ in range(n_units):
        model.add(Dense(n_hidden, activation="relu"))
    model.add(Dense(10))

    return model

def simple_Conv_NL(n_hidden,kernel_size=10):
    """ no lambda """
    model = Sequential()
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10))

    return model

print("Defining shift functions")

def shift_images(images, shift, axis=0):
    shifted_images = np.roll(images, shift, axis=axis)
    return shifted_images

def evaluate_shift_invariance(model, x_test, y_test, shifts, axis=0):
    accuracies = []
    for shift in shifts:
        shifted_images = shift_images(x_test, shift, axis=axis)
        predictions = model.predict(shifted_images, verbose=2)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        accuracies.append(accuracy)
    return accuracies


def train_and_evaluate_models(models, model_names, x_train, y_train, x_test, y_test):
    accuracy_data = {}
    shifts = range(10, -11, -1)

    for model, model_name in  tzip(models, model_names):
        loss_ = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss_, metrics=['accuracy'])
        tensorboard = TensorBoard(log_dir=f"logs/{model_name}")
        print("training: ", model_name)
        model.fit(x_train, y_train, batch_size=256, epochs=10, callbacks=[tensorboard], verbose=0) #  validation_data=(x_test, y_test),
        accuracy_ = evaluate_shift_invariance(model, x_test, y_test, shifts, axis=1)
        accuracy_data[model_name] = accuracy_

    return accuracy_data

print("Creating models")
models_ = []
model_names_ = []

model_FC_2_256 = simple_FC(256, 2)
models_.append(model_FC_2_256)
model_names_.append('simple_FC_2_256')
model_FC_3_256 = simple_FC(256, 3)
models_.append(model_FC_3_256)
model_names_.append('simple_FC_3_256')
model_Conv_NL = simple_Conv_NL(512)
models_.append(model_Conv_NL)
model_names_.append('simple_Conv_NL')
padding_sizes = [0, 2, 4, 6, 8, 10, 14]
n_hidden_values = [512, 2048]

for padding_size in padding_sizes:
    for n_hidden in n_hidden_values:
        model_name = f"simple_Conv_{padding_size}_{n_hidden}"
        model = simple_Conv(n_hidden, kernel_size=12, padding_size=padding_size)
        models_.append(model)
        model_names_.append(model_name)


print("MAIN FUNCTION")
accuracy_data = train_and_evaluate_models(models_, model_names_, x_train, y_train, x_test, y_test)
print("training and evaluation done")


experiment_dir = "padding_experiment"
os.makedirs(experiment_dir, exist_ok=True)

print("Saving Accuracy data")
accuracy_data_path = os.path.join(experiment_dir, "accuracy_data.pkl")
with open(accuracy_data_path, "wb") as f:
    pickle.dump(accuracy_data, f)

print("Saving models")
for i, model in enumerate(models_):
    model_name = model_names_[i]
    model_path = os.path.join(experiment_dir, f"{model_name}.h5")
    model.save(model_path)

print("Done!")