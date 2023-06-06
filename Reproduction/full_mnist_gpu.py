import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, GlobalAveragePooling2D, Dense, Lambda
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentTensorFlowV2
from tqdm import tqdm
import os
from art.utils import load_mnist

# Set the number of GPUs to use


# Set GPU devices
physical_devices = tf.config.list_physical_devices("GPU")
print("gpu", len(physical_devices))
num_gpus = len(physical_devices)
if num_gpus>0:
    tf.config.experimental.set_visible_devices(physical_devices[:num_gpus], "GPU")
    logical_devices = tf.config.experimental.list_logical_devices("GPU")
    tf.distribute.OneDeviceStrategy(logical_devices[0])
    tf.distribute.MirroredStrategy(logical_devices)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

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

n_hidden = 1000
padding_sizes = [0, 2, 4, 6, 8, 10, 12, 14]
padding_size = padding_sizes[5]

@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Create the CNN model and optimizer
model = simple_Conv(n_hidden, kernel_size=28, padding_size=padding_size)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.1, momentum=0.9, decay=5e-4)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Create the ART classifier
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=10,
        input_shape=(28, 28, 1),
        clip_values=(0, 1),
    )

# Fit the classifier to the training data
classifier_history = classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3, verbose=2)

# Perform predictions and evaluate accuracy on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

import numpy as np
from tqdm import tqdm
import os
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentTensorFlowV2

# Define attack parameters
attack_params = [[np.inf, [0.05, 0.1, 0.15, 0.2, 0.25]] , [2, [0.5, 1, 1.5, 2, 2.5]]]

# Iterate over the attack parameters and generate adversarial examples
for norm, epsilons in attack_params:
    for epsilon in epsilons:
        if norm == 2:
            attack = FastGradientMethod(estimator=classifier, eps=epsilon, norm=norm)
        else:
            attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=epsilon, norm=norm)

        attack_name = attack.__class__.__name__
        model_name = "simple_Conv_28_10_1000"

        adv_correct = 0
        adv_loss = 0
        total = 0
        x_train_attack = []
        y_train_attack = []
        x_test_attack = []
        y_test_attack = []

        x_train_attack = attack.generate(x=x_train)
        y_train_attack = np.copy(y_train)

        x_test_attack = attack.generate(x=x_test)
        y_test_attack = np.copy(y_test)

        x_train_attack = np.array(x_train_attack)
        y_train_attack = np.array(y_train_attack)
        x_test_attack = np.array(x_test_attack)
        y_test_attack = np.array(y_test_attack)

        save_dir = "adversarial_data"
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, f"{model_name}_{attack_name}_{epsilon}_train.npz"),
                 x_train_attack=x_train_attack, y_train_attack=y_train_attack)
        np.savez(os.path.join(save_dir, f"{model_name}_{attack_name}_{epsilon}_test.npz"),
                 x_test_attack=x_test_attack, y_test_attack=y_test_attack)

        print(os.path.join(save_dir, f"{model_name}_{attack_name}_{epsilon}"))

        for x, y in tqdm(zip(x_test_attack, y_test_attack), total=len(y_test_attack),
                         desc="Evaluating Adversarial Examples"):
            predictions_adv = np.argmax(classifier.predict(np.expand_dims(x, axis=0)), axis=1)
            adv_correct += (predictions_adv == y).sum()
            total += 1

        _, adv_loss = classifier.model.evaluate(x_test_attack, y_test_attack, verbose=0)
        accuracy = adv_correct / total
        print("Accuracy on adversarial test examples (L_{:.0f}, eps={:.2f}): {:.2f}%. Loss: {:.2f}".format(
            norm, epsilon, accuracy * 100, adv_loss))