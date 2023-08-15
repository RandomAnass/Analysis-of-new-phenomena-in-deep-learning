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
""" changed max_iter to 100 and epochs to 20 and data to max instead of 1/6 """
print("seed: 420")
master_seed(420)
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

# Load MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()



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
# 512


def train_step(model, optimizer, loss_object, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# nb_epochs=20
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



n_hidden = 256
padding_size = 5

n_layers_list = [1, 2, 3]
add_dense_list = [False]
max_pooling_list = [True]
kernel_size_list = [3, 10]

models = []
model_names = {}

model_names['simple_FC_256']=simple_FC
models.append(create_art_classifier(model_creator=simple_FC, x_train=x_train, y_train=y_train,  x_test=x_test, y_test=y_test, n_hidden=n_hidden))
for n_layers in n_layers_list:
    for add_dense in add_dense_list:
        for max_pooling in max_pooling_list:
            for kernel_size in kernel_size_list:
                model_name = f"epsilon_simple_Conv_{n_layers}_{max_pooling}_{kernel_size}_{add_dense}"
                print("Training: ", model_name)
                model_names[model_name] = simple_Conv
                model = create_art_classifier(
                    model_creator=simple_Conv,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    n_hidden=n_hidden,
                    kernel_size=kernel_size,
                    padding_size=padding_size,
                    n_layers=n_layers,
                    max_pooling=max_pooling,
                    add_dense=add_dense
                )
                models.append(model)


print("Done training")


array_models = np.array(models)
acc_array = array_models[:, [1,2]]
loss_array = array_models[:, [3,4]]



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

attack_params = [[2, [2, 5,  10, 25]]]
#attack_params = [[np.inf, [0.05, 0.1,  0.15, 0.2, 0.25, 0.3]],[2, [0.5, 1, 1.5,  2.5, 3]]]

save_dir = "extension_mnist_epsilon"
# Define attack parameters
os.makedirs(save_dir, exist_ok=True)


norms = [attack_params[0][0]]
epsilons = {
    attack_params[0][0]: attack_params[0][1]
}

accuracy_data = {norm: {model: [] for model in list(model_names.keys())} for norm in norms}


print("Creating attack data")
for model, model_name in zip(models, model_names.keys()):
    classifier = model[0]
    # Iterate over the attack parameters and generate adversarial examples
    for norm, epsilons in attack_params:
        # Calculate accuracy on normal data (epsilon = 0)
        predictions = classifier.predict(x_test, verbose=0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        accuracy_data[norm][model_name].append((0, accuracy))
        for epsilon in epsilons:
            print( f"{model_name}_{epsilon}")
            attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=epsilon, batch_size=100, norm=norm)


            attack_name = attack.__class__.__name__




            x_test_attack = attack.generate(x=x_test)
            y_test_attack = np.copy(y_test)

            x_test_attack = np.array(x_test_attack)





            # Calculate accuracy on adversarial data (epsilon > 0)
            predictions = classifier.predict(x_test_attack, verbose=0)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_attack, axis=1)) / len(y_test_attack)
            accuracy_data[norm][model_name].append((epsilon, accuracy))

print("Done Attack data created")
# Save the accuracy data
with open('epsilon_mnist_accuracy_data.pkl', 'wb') as file:
    pickle.dump(accuracy_data, file)

# Save the model names
with open('epsilon_mnist_model_names.pkl', 'wb') as file:
    pickle.dump(model_names, file)

print("Done creating attack data")


