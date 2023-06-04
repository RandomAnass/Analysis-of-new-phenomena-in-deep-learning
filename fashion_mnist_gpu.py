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
print("data loaded"

# Normalize the pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the input data to add a channel dimension (needed for convolutional models)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert the labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
print(tf.test.gpu_device_name())
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.list_physical_devices("GPU")
print("gpus:", physical_devices)

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

def simple_Conv_NL(n_hidden,kernel_size=28):
    """ no lambda """
    model = Sequential()
    model.add(Conv2D(n_hidden, kernel_size=kernel_size, padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10))

    return model

def simple_FC(n_hidden):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
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
               'simple_Conv_max':simple_Conv_max, 'simple__RNN':simple__RNN , 'simple_FC_2':simple_FC_2, 'simple_Conv_2':simple_Conv_2

padding_size = 0 #padding_sizes[3]
n_hidden = 1024 #1000
kernel_size=28


def train_step(model, optimizer, loss_object, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def create_art_classifier(model_creator, x_train, y_train, batch_size=200, nb_epochs=30, **kwargs): # nb_epochs=200
    # Create the CNN model and optimizer
    model = model_creator(**kwargs)
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.1, momentum=0.9, decay=5e-4)
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
    # Custom training loop
    train_acc = []
    test_acc = []
    for epoch in range(nb_epochs):
        print("Epoch {}/{}".format(epoch + 1, nb_epochs))
        epoch_loss = []
        for batch in range(0, len(x_train), batch_size):
            batch_images = x_train[batch:batch + batch_size]
            batch_labels = y_train[batch:batch + batch_size]
            loss = train_step(model, optimizer, loss_object, batch_images, batch_labels)
            epoch_loss.append(loss)
        #print(epoch_loss)
        avg_loss = np.mean(epoch_loss)
        print("Average loss: {:.4f}".format(avg_loss))

        # Perform predictions and evaluate accuracy on examples
        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        test_acc.append(accuracy)
        print("Accuracy on test examples: {:.2%}".format(accuracy))

        predictions = classifier.predict(x_train)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / len(y_train)
        train_acc.append(accuracy)
        print("Accuracy on train examples: {:.2%}".format(accuracy))

    return [classifier, test_acc, train_acc]

models = [
    create_art_classifier(model_creator=simple_FC, x_train=x_train, y_train=y_train, n_hidden=n_hidden),
    create_art_classifier(model_creator=simple_Conv, x_train=x_train, y_train=y_train, n_hidden=n_hidden, kernel_size=kernel_size, padding_size=padding_size),
    create_art_classifier(model_creator=simple_Conv_NL, x_train=x_train, y_train=y_train, n_hidden=n_hidden, kernel_size=kernel_size),
    create_art_classifier(model_creator=simple_Conv_max, x_train=x_train, y_train=y_train, n_hidden=n_hidden, kernel_size=kernel_size),
    create_art_classifier(model_creator=simple__RNN, x_train=x_train, y_train=y_train),
    create_art_classifier(model_creator=simple_FC_2, x_train=x_train, y_train=y_train),
    create_art_classifier(model_creator=simple_Conv_2, x_train=x_train, y_train=y_train)
]
for i, model in enumerate(models):
    model_name = f'model_{i}.h5'
    model[0].model.save(model_name)

array_models = np.array(models)
acc_array = array_models[:, [1,2]]


# Save the models list to a file
with open('acc_array.pkl', 'wb') as file:
    pickle.dump(acc_array, file)

with open('full_np.pkl', 'wb') as file:
    pickle.dump(array_models, file)
print("done saving models")
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

            x_train_attack = attack.generate(x=x_train[:3000])
            y_train_attack = np.copy(y_train[:3000])

            x_test_attack = attack.generate(x=x_test[:3000])
            y_test_attack = np.copy(y_test[:3000])

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