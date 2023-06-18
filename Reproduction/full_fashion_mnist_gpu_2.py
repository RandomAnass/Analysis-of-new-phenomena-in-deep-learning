import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, GlobalAveragePooling2D,MaxPooling2D, Dense, Lambda, Flatten, GlobalMaxPooling2D,Flatten, Reshape, LSTM
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentTensorFlowV2
from tqdm import tqdm
import os
from art.utils import load_mnist
from art.estimators.classification import TensorFlowV2Classifier
import pickle


# Set GPU devices
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

print("data processed")

def circular_padding(x, padding_size):
    # Perform circular padding on the input tensor
    return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='SYMMETRIC')

def simple_Conv(n_hidden, kernel_size=10, padding_size=-1):
    if padding_size == -1:
        padding_size = kernel_size // 2

    model = Sequential()
    model.add(Lambda(lambda x: circular_padding(x, padding_size), input_shape=(28, 28, 1)))
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='same'))
    model.add(ReLU())
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

def simple_Conv_NL(n_hidden,kernel_size=10):
    """ no lambda """
    model = Sequential()
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10))

    return model

def simple_Conv_max(n_hidden, kernel_size=10,padding_size=0):
    model = Sequential()
    model.add(Lambda(lambda x: circular_padding(x, padding_size), input_shape=(28, 28, 1)))
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(10))

    return model

def simple_Conv_2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10))

    return model

def simple_FC_2():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))

    return model


def simple__RNN():
    model = Sequential()
    model.add(Reshape((28, 28), input_shape=(28, 28, 1)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(10))

    return model


n_hidden = 1000
padding_sizes = [0, 2, 4, 6, 8, 10, 12, 14]
padding_size = padding_sizes[5]


def train_step(model, optimizer, loss_object, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def create_art_classifier(model_creator, x_train, y_train, x_test, y_test, batch_size=200, nb_epochs=20, **kwargs):
    model = model_creator(**kwargs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])

    # Create the ART classifier
    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=10,
        input_shape=(28, 28, 1),
        clip_values=(0, 1),
    )

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    for epoch in range(nb_epochs):
        print("Epoch {}/{}".format(epoch + 1, nb_epochs))
        epoch_loss = []

        if epoch == 10 or epoch == 15:
            optimizer.learning_rate.assign(optimizer.learning_rate * 0.1)

        for batch in range(0, len(x_train), batch_size):
            batch_images = x_train[batch:batch + batch_size]
            batch_labels = y_train[batch:batch + batch_size]
            loss = train_step(model, optimizer, loss_object, batch_images, batch_labels)
            epoch_loss.append(loss)

        avg_loss = np.mean(epoch_loss)
        train_loss.append(avg_loss)
        print("Average training loss: {:.4f}".format(avg_loss))

        # evaluate accuracy on test examples
        predictions = classifier.predict(x_test)
        test_loss_value = loss_object(y_test, predictions).numpy()
        test_loss.append(test_loss_value)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        test_acc.append(accuracy)
        print("Accuracy on test examples: {:.2%}".format(accuracy))

        # evaluate accuracy on train examples
        predictions = classifier.predict(x_train)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / len(y_train)
        train_acc.append(accuracy)
        print("Accuracy on train examples: {:.2%}".format(accuracy))

    return [classifier, test_acc, train_acc, test_loss, train_loss]

print("creating models")
model_names = {'simple_FC_2_256':simple_FC, 'simple_Conv_10_512':simple_Conv, 'simple_Conv_NL':simple_Conv_NL,
               'simple_Conv_max':simple_Conv_max, 'simple_FC_3_512':simple_FC, 'simple_Conv_12_2048':simple_Conv,
               'simple__RNN':simple__RNN , 'simple_FC_2':simple_FC_2, 'simple_Conv_2':simple_Conv_2}

models = [
    create_art_classifier(model_creator=simple_FC, x_train=x_train, y_train=y_train,  x_test=x_test, y_test=y_test, n_hidden=256,n_units=2),
    create_art_classifier(model_creator=simple_Conv, x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, n_hidden=512, kernel_size=10, padding_size=padding_size),
    create_art_classifier(model_creator=simple_Conv_NL, x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, n_hidden=512, kernel_size=10),
    create_art_classifier(model_creator=simple_Conv_max, x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, n_hidden=512, kernel_size=10),
    create_art_classifier(model_creator=simple_FC, x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, n_hidden=512,n_units=3),
    create_art_classifier(model_creator=simple_Conv, x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, n_hidden=2048, kernel_size=12, padding_size=padding_size),
    create_art_classifier(model_creator=simple__RNN, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
    create_art_classifier(model_creator=simple_FC_2, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
    create_art_classifier(model_creator=simple_Conv_2, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
]


print("saving the models")
print("Saving the models")
for i, model in enumerate(models):
    model_name = list(model_names.keys())[i]
    model[0].model.save("fashion_mnist_" +model_name + ".h5")


array_models = np.array(models)
acc_array = array_models[:, [1,2]]
loss_array = array_models[:, [3,4]]

# Save the models list to a file
with open('mnist_acc_array.pkl', 'wb') as file:
    pickle.dump(acc_array, file)
with open('mnist_loss_array.pkl', 'wb') as file:
    pickle.dump(loss_array, file)
with open('mnist_full_np.pkl', 'wb') as file:
    pickle.dump(array_models, file)
print("done saving models")


print("creating adversarial data")
import numpy as np
from tqdm import tqdm
import os
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentTensorFlowV2
print("ART attacks")
# Enable GPU acceleration for ART attacks
from art.config import ART_NUMPY_DTYPE
os.environ["ART_NUMPY_DTYPE"] = str(ART_NUMPY_DTYPE)
# Enable GPU acceleration for ART attacks
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Define the attack parameters
attack_params = [[np.inf, [0.05, 0.1,  0.15, 0.2, 0.25, 0.3]],[2, [0.5, 1, 1.5,  2.5, 3]]]
save_dir = "fashion_adversarial_data_2"
# Define attack parameters
os.makedirs(save_dir, exist_ok=True)
print("creating attack data_2")

for model, model_name in zip(models, model_names.keys()):
    classifier = model[0]
    # Iterate over the attack parameters and generate adversarial examples
    for norm, epsilons in attack_params:
        for epsilon in epsilons:
            if norm == 2:
                attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=epsilon,eps_step=epsilon/5,max_iter=5, norm=norm)
            else:
                attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=epsilon,eps_step=epsilon/5,max_iter=5, norm=norm)

            attack_name = attack.__class__.__name__

            file_path_train = os.path.join(save_dir, f"2_{model_name}_{attack_name}_{epsilon}_train.npz")
            file_path_test = os.path.join(save_dir, f"2_{model_name}_{attack_name}_{epsilon}_test.npz")

            if os.path.exists(file_path_train) and os.path.exists(file_path_test):
                print(f"Skipping creation of {file_path_train} and {file_path_test}")
                continue

            x_train_attack = attack.generate(x=x_train[:10000])
            y_train_attack = np.copy(y_train[:10000])
            x_test_attack = attack.generate(x=x_test[:5000])
            y_test_attack = np.copy(y_test[:5000])

            x_train_attack = np.array(x_train_attack)
            #y_train_attack = np.array(y_train_attack)
            x_test_attack = np.array(x_test_attack)
            #y_test_attack = np.array(y_test_attack)

            np.savez(file_path_train, x_train_attack=x_train_attack, y_train_attack=y_train_attack)
            np.savez(file_path_test, x_test_attack=x_test_attack, y_test_attack=y_test_attack)

            print(os.path.join(save_dir, f"{model_name}_{attack_name}_{epsilon}"))

print("Done")