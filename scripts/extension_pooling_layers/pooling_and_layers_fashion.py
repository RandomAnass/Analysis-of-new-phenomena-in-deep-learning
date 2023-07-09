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

# Load fashion_mnist dataset
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

print("testing mode?")
testing_mode = str(input("testing mode ? T or F"))
print("given answer: ", testing_mode)
if testing_mode == "T":
    x_train = x_train[:20]
    x_test = x_test[:20]
    y_train = y_test[:20]
    y_test = y_test[:20]
elif testing_mode == "F":
    print("Training mode")
else:
    print("Error: Answer must be T or F")



##################################################
print("defining functions")

def simple_FC(n_hidden, n_units = 2):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for i in range(n_units):
        model.add(Dense(n_hidden, activation="relu"))
    model.add(Dense(10))

    return model
#    create_art_classifier(model_creator=simple_FC, x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, n_hidden=512,n_units=3),

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


def evaluate_shift_consistency(models,model_names, x_test,max_shift=10):
    """ Evaluate shift consistency of models
    we can use predict directly but predict_on_batch is much faster, except it requires more memory"""
    model_consistency_dict = {}

    for model_name, model in zip(model_names, models):
        print(model_name)
        model_consistency = 0
        model_total = 0

        shift0 = np.random.randint(max_shift, size=(len(x_test), 2))
        shift1 = np.random.randint(max_shift, size=(len(x_test), 2))

        inputs_shift_v0 = np.zeros_like(x_test)
        inputs_shift_hv0 = np.zeros_like(x_test)
        inputs_shift_v1 = np.zeros_like(x_test)
        inputs_shift_hv1 = np.zeros_like(x_test)

        for i in range(len(x_test)):
            image = x_test[i]

            inputs_shift_v0[i, :, :shift0[i, 0], :] = image[:, (image.shape[1] - shift0[i, 0]):, :].copy()
            inputs_shift_v0[i, :, shift0[i, 0]:, :] = image[:, :(image.shape[1] - shift0[i, 0]), :].copy()
            inputs_shift_hv0[i, :shift0[i, 1], :, :] = inputs_shift_v0[i, (inputs_shift_v0.shape[1] - shift0[i, 1]):, :, :]
            inputs_shift_hv0[i, shift0[i, 1]:, :, :] = inputs_shift_v0[i, :(inputs_shift_v0.shape[1] - shift0[i, 1]), :, :]

            inputs_shift_v1[i, :, :shift1[i, 0], :] = image[:, (image.shape[1] - shift1[i, 0]):, :].copy()
            inputs_shift_v1[i, :, shift1[i, 0]:, :] = image[:, :(image.shape[1] - shift1[i, 0]), :].copy()
            inputs_shift_hv1[i, :shift1[i, 1], :, :] = inputs_shift_v1[i, (inputs_shift_v1.shape[1] - shift1[i, 1]):, :, :]
            inputs_shift_hv1[i, shift1[i, 1]:, :, :] = inputs_shift_v1[i, :(inputs_shift_v1.shape[1] - shift1[i, 1]), :, :]

        batch_size = 100  # Adjust the batch size as needed
        num_batches = int(np.ceil(len(x_test) / batch_size))

        predicted0_list = []
        predicted1_list = []

        for j in tqdm(range(num_batches)):
            start_idx = j * batch_size
            end_idx = (j + 1) * batch_size

            predicted0_batch = np.argmax(model[0].predict_on_batch(inputs_shift_hv0[start_idx:end_idx]), axis=1)
            predicted1_batch = np.argmax(model[0].predict_on_batch(inputs_shift_hv1[start_idx:end_idx]), axis=1)

            predicted0_list.append(predicted0_batch)
            predicted1_list.append(predicted1_batch)

        predicted0 = np.concatenate(predicted0_list)
        predicted1 = np.concatenate(predicted1_list)

        model_consistency = np.sum(predicted0 == predicted1) / len(x_test)
        model_consistency_dict[model_name] = model_consistency

    return model_consistency_dict
##################################################

print("params")
# Enable GPU acceleration for ART attacks
from art.config import ART_NUMPY_DTYPE
os.environ["ART_NUMPY_DTYPE"] = str(ART_NUMPY_DTYPE)
# Enable GPU acceleration for ART attacks
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Define the attack parameters
attack_params = [[np.inf, [0.05, 0.1,  0.15, 0.2]],[2, [0.5, 1,  1.5, 2]]]
#attack_params = [[np.inf, [0.05, 0.1,  0.15, 0.2, 0.25, 0.3]],[2, [0.5, 1, 1.5,  2.5, 3]]]

save_dir = "extension_pooling_layers"
# Define attack parameters
os.makedirs(save_dir, exist_ok=True)


norms = [attack_params[0][0], attack_params[1][0]]
epsilons = {
    attack_params[0][0]: attack_params[0][1],
    attack_params[1][0]: attack_params[1][1]
}




n_hidden = 256
padding_size = 5

n_layers_list = [1, 2, 3, 4]
add_dense_list = [False]
max_pooling_list = [True]
kernel_size_list = [2, 3, 6, 10, 14, 20] # 4*6=24

models = []
model_names = {}
model_names['simple_FC_2_256']=simple_FC
model_names['simple_FC_3_512']=simple_FC
print("FC models")
for n_layers in n_layers_list:
    for add_dense in add_dense_list:
        for max_pooling in max_pooling_list:
            for kernel_size in kernel_size_list:
                model_name = f"simple_Conv_{n_layers}_{max_pooling}_{kernel_size}_{add_dense}"
                model_names[model_name] = simple_Conv
accuracy_data = {norm: {model: [] for model in list(model_names.keys())} for norm in norms}


model_name = 'simple_FC_2_256'
model = create_art_classifier(model_creator=simple_FC, x_train=x_train, y_train=y_train,  x_test=x_test, y_test=y_test, n_hidden=256,n_units=2)
classifier = model[0]
for norm, epsilons in attack_params:
    # Calculate accuracy on normal data (epsilon = 0)
    predictions = classifier.predict(x_test, verbose=0)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    accuracy_data[norm][model_name].append((0, accuracy))
    for epsilon in epsilons:

        attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=epsilon, max_iter=100, batch_size=100, norm=norm) #changed to 100 # changed eps_step=epsilon/5
        attack_name = attack.__class__.__name__
        print(f"{model_name}_{attack_name}_{epsilon}")
        x_test_attack = attack.generate(x=x_test)
        y_test_attack = np.copy(y_test)
        x_test_attack = np.array(x_test_attack)
        # Calculate accuracy on adversarial data (epsilon > 0)
        predictions = classifier.predict(x_test_attack, verbose=0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_attack, axis=1)) / len(y_test_attack)
        accuracy_data[norm][model_name].append((epsilon, accuracy))
models.append(model)

model_name = 'simple_FC_3_512'
model = create_art_classifier(model_creator=simple_FC, x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, n_hidden=512,n_units=3)
classifier = model[0]
for norm, epsilons in attack_params:
    # Calculate accuracy on normal data (epsilon = 0)
    predictions = classifier.predict(x_test, verbose=0)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    accuracy_data[norm][model_name].append((0, accuracy))
    for epsilon in epsilons:

        attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=epsilon, max_iter=100, batch_size=100, norm=norm) #changed to 100 # changed eps_step=epsilon/5
        attack_name = attack.__class__.__name__
        print(f"{model_name}_{attack_name}_{epsilon}")
        x_test_attack = attack.generate(x=x_test)
        y_test_attack = np.copy(y_test)
        x_test_attack = np.array(x_test_attack)
        # Calculate accuracy on adversarial data (epsilon > 0)
        predictions = classifier.predict(x_test_attack, verbose=0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_attack, axis=1)) / len(y_test_attack)
        accuracy_data[norm][model_name].append((epsilon, accuracy))
models.append(model)


print("CNN models")

for n_layers in n_layers_list:
    for add_dense in add_dense_list:
        for max_pooling in max_pooling_list:
            for kernel_size in kernel_size_list:
                model_name = f"simple_Conv_{n_layers}_{max_pooling}_{kernel_size}_{add_dense}"
                print("Training: ", model_name)
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
                classifier = model[0]
                for norm, epsilons in attack_params:
                    # Calculate accuracy on normal data (epsilon = 0)
                    predictions = classifier.predict(x_test, verbose=0)
                    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
                    accuracy_data[norm][model_name].append((0, accuracy))
                    for epsilon in epsilons:

                        attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=epsilon, max_iter=100, batch_size=100, norm=norm) #changed to 100 # changed eps_step=epsilon/5
                        attack_name = attack.__class__.__name__
                        print(f"{model_name}_{attack_name}_{epsilon}")
                        x_test_attack = attack.generate(x=x_test)
                        y_test_attack = np.copy(y_test)
                        x_test_attack = np.array(x_test_attack)
                        # Calculate accuracy on adversarial data (epsilon > 0)
                        predictions = classifier.predict(x_test_attack, verbose=0)
                        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_attack, axis=1)) / len(y_test_attack)
                        accuracy_data[norm][model_name].append((epsilon, accuracy))
                models.append(model)
                tf.keras.backend.clear_session()



print("Saving the models")
for i, model in enumerate(models):
    model_name = f'{list(model_names.keys())[i]}.h5'
    model[0].model.save(model_name)

array_models = np.array(models)
acc_array = array_models[:, [1,2]]
loss_array = array_models[:, [3,4]]

# Save the models list to a file
with open('fashion_mnist_acc_array.pkl', 'wb') as file:
    pickle.dump(acc_array, file)
with open('fashion_mnist_loss_array.pkl', 'wb') as file:
    pickle.dump(loss_array, file)
with open('fashion_mnist_full_np.pkl', 'wb') as file:
    pickle.dump(array_models, file)
print("done saving models")





print("Done Attack data created")
# Save the accuracy data
with open('fashion_mnist_accuracy_data.pkl', 'wb') as file:
    pickle.dump(accuracy_data, file)

# Save the model names
with open('fashion_mnist_model_names.pkl', 'wb') as file:
    pickle.dump(model_names, file)

print("Done creating attack data")



print("starting consistency")



model_consistency_dict = evaluate_shift_consistency(models,list(model_names.keys()), x_test)

print("saving model_consistency_dict")
# Save the model_consistency_dict
with open('fashion_mnist_model_consistency_dict.pkl', 'wb') as file:
    pickle.dump(model_consistency_dict, file)

print("Done")