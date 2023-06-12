import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, GlobalAveragePooling2D, Dense, Lambda, GlobalMaxPooling2D,Flatten
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentTensorFlowV2
#from tqdm import tqdm
import os
from tensorflow.keras import layers
from art.estimators.classification import TensorFlowV2Classifier
import pickle

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("data loaded")

# Normalize the pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the input data to add a channel dimension (needed for convolutional models)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert the labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

#print(tf.test.gpu_device_name())

#ValueError: Memory growth cannot differ between GPU devices
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
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.list_physical_devices("GPU")
#print("gpus:", physical_devices)

def circular_padding(x, padding_size):
    # Perform circular padding on the input tensor
    return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='SYMMETRIC')

def simple_Conv(n_hidden, kernel_size=10, padding_size=-1):
    if padding_size == -1:
        padding_size = kernel_size // 2

    model = Sequential()
    model.add(Lambda(lambda x: circular_padding(x, padding_size), input_shape=(28, 28, 1)))
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='valid'))
    model.add(ReLU())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10))

    return model

def simple_Conv_NL(n_hidden,kernel_size=10):
    """ no lambda """
    model = Sequential()
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10))

    return model

def simple_FC(n_hidden, n_units = 2):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for i in range(n_units):
        model.add(Dense(n_hidden, activation="relu"))
    model.add(Dense(10))

    return model

def simple_Conv_max(n_hidden, kernel_size=28):
    model = Sequential()
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(10))

    return model

def simple_Conv_max(n_hidden, kernel_size=28):
    model = Sequential()
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(10))

    return model

def simple_Conv_2():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model

def simple_FC_2():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model


def simple__RNN():

    model = tf.keras.Sequential([
        layers.Reshape((28, 28), input_shape=(28, 28, 1)),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(128),
        layers.Dense(10, activation='softmax')
    ])

    return model

print("creating models")
model_names = {'simple_FC':simple_FC, 'simple_Conv':simple_Conv, 'simple_Conv_NL':simple_Conv_NL,
               'simple_Conv_max':simple_Conv_max, 'simple_FC_':simple_FC, 'simple_Conv_':simple_Conv,
               'simple__RNN':simple__RNN , 'simple_FC_2':simple_FC_2, 'simple_Conv_2':simple_Conv_2}

padding_size = 5 #padding_sizes[3]
n_hidden = 1024 #1000
kernel_size=28

##########
print("loading models")
LRZ = True
if LRZ:
    base_path = "/dss/dsshome1/lxc0C/apdl010/Analysis-of-new-phenomena-in-deep-learning/Reproduction"
else:
    base_path =  "C:/Users/anass/Desktop/TUM/Courses/Practical/Github/Practical-Course---Analysis-of-new-phenomena-in-machine-deep-learning/data and models/models/fashion_mnist"

simple_FC = tf.keras.models.load_model(base_path + '/model_0.h5')
simple_Conv = tf.keras.models.load_model(base_path+'/model_1.h5')
simple_Conv_NL = tf.keras.models.load_model(base_path+'/model_2.h5')
simple_Conv_max   = tf.keras.models.load_model(base_path+'/model_3.h5')
simple_FC_ = tf.keras.models.load_model(base_path+'/model_4.h5')
simple_Conv_ = tf.keras.models.load_model(base_path+'/model_5.h5')
simple__RNN =  tf.keras.models.load_model(base_path+'/model_6.h5')
simple_FC_2 =  tf.keras.models.load_model(base_path+'/model_7.h5')
simple_Conv_2 =  tf.keras.models.load_model(base_path+'/model_8.h5')

with open(base_path + '/acc_array.pkl', 'rb') as file:
    acc_array = pickle.load(file)
with open(base_path + '/loss_array.pkl', 'rb') as file:
    loss_array = pickle.load(file)


models = [simple_FC, simple_Conv, simple_Conv_NL, simple_Conv_max, simple_FC_,simple_Conv_, simple__RNN,simple_FC_2,simple_Conv_2]
models = [[models[i]] + [acc_array[i][0]] + [acc_array[i][1]] + [loss_array[i][0]] + [loss_array[i][1]] for i in range(len(models))]
############
print("ART attacks")
# Enable GPU acceleration for ART attacks
from art.config import ART_NUMPY_DTYPE
os.environ["ART_NUMPY_DTYPE"] = str(ART_NUMPY_DTYPE)

# Define the attack parameters
attack_params = [[np.inf, [0.05, 0.1,  0.15, 0.2, 0.25, 0.3]],[2, [0.5, 1, 1.5,  2.5, 3]]]
save_dir = "adversarial_fashion_data"
os.makedirs(save_dir, exist_ok=True)
print("creating attack data")
for model, model_name in zip(models, model_names.keys()):
    classifier = model[0]
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    classifier = TensorFlowV2Classifier(
        model=classifier,
        loss_object=loss_object,
        #train_step=train_step,
        nb_classes=10,
        input_shape=(28, 28, 1),
        clip_values=(0, 1),
    )
    # Iterate over the attack parameters and generate adversarial examples
    for norm, epsilons in attack_params:
        for epsilon in epsilons:
            if norm == 2:
                attack = FastGradientMethod(estimator=classifier, eps=epsilon, norm=norm)
            else:
                attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=epsilon, norm=norm)

            attack_name = attack.__class__.__name__
            #model_name = "simple_Conv_28_10_1000"

            adv_correct = 0
            adv_loss = 0
            total = 0
            x_train_attack = []
            y_train_attack = []
            x_test_attack = []
            y_test_attack = []

            x_train_attack = attack.generate(x=x_train[:10000])
            y_train_attack = np.copy(y_train[:10000])

            x_test_attack = attack.generate(x=x_test[:5000])
            y_test_attack = np.copy(y_test[:5000]) # [:3000]

            x_train_attack = np.array(x_train_attack)
            #y_train_attack = np.array(y_train_attack)
            x_test_attack = np.array(x_test_attack)
            #y_test_attack = np.array(y_test_attack)





            np.savez(os.path.join(save_dir, f"{model_name}_{attack_name}_{epsilon}_train.npz"),
                     x_train_attack=x_train_attack, y_train_attack=y_train_attack)
            np.savez(os.path.join(save_dir, f"{model_name}_{attack_name}_{epsilon}_test.npz"),
                     x_test_attack=x_test_attack, y_test_attack=y_test_attack)

            print(os.path.join(save_dir, f"{model_name}_{attack_name}_{epsilon}"))

print("Done")