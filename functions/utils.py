import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def evaluate_shift_consistency_memory(models, x_test):
    """this function can be used instead of evaluate_shift_consistency if enough memory isnt available to use predict_on_batch
    it's much slower but it doesnt use a lot of memory"""
    model_consistency_dict = {}

    for (model_name, _), model in zip(model_names.items(), models):
        model_consistency = 0
        model_total = 0

        for _ in tqdm(range(len(x_test))):  # range(5):
            index = np.random.randint(len(x_test))
            image = x_test[index]

            shift0 = np.random.randint(10, size=2)
            inputs_shift_v0 = np.zeros([1, 28, 28, 1])
            inputs_shift_hv0 = np.zeros([1, 28, 28, 1])
            inputs_shift_v0[:, :, :shift0[0], :] = image[:, (28 - shift0[0]):, :].copy()
            inputs_shift_v0[:, :, shift0[0]:, :] = image[:, :(28 - shift0[0]), :].copy()
            inputs_shift_hv0[:, :shift0[1], :, :] = inputs_shift_v0[:, (28 - shift0[1]):, :, :]
            inputs_shift_hv0[:, shift0[1]:, :, :] = inputs_shift_v0[:, :(28 - shift0[1]), :, :]

            shift1 = np.random.randint(10, size=2)
            inputs_shift_v1 = np.zeros([1, 28, 28, 1])
            inputs_shift_hv1 = np.zeros([1, 28, 28, 1])
            inputs_shift_v1[:, :, :shift1[0], :] = image[:, (28 - shift1[0]):, :].copy()
            inputs_shift_v1[:, :, shift1[0]:, :] = image[:, :(28 - shift1[0]), :].copy()
            inputs_shift_hv1[:, :shift1[1], :, :] = inputs_shift_v1[:, (28 - shift1[1]):, :, :]
            inputs_shift_hv1[:, shift1[1]:, :, :] = inputs_shift_v1[:, :(28 - shift1[1]), :, :]

            predicted0 = np.argmax(model[0].predict(inputs_shift_hv0, verbose=0), axis=1)
            predicted1 = np.argmax(model[0].predict(inputs_shift_hv1, verbose=0), axis=1)

            model_consistency += np.sum(predicted0 == predicted1)
            model_total += 1

        model_consistency /= model_total
        model_consistency_dict[model_name] = model_consistency

    return model_consistency_dict

def extract_padding(model_name):
    if 'simple_Conv' in model_name and 'NL' not in model_name:
        return int(model_name.split('_')[2])
    else:
        return 0

def extract_layer_size(model_name):
    return int(model_name.split('_')[3])

def plot_shift_examples(x_test,n=2):
    for _ in range(n): #:
        index = np.random.randint(len(x_test))
        image = x_test[index]
        shift0 = np.random.randint(7, size=2)
        inputs_shift_v0 = np.zeros([1, 28, 28, 1])
        inputs_shift_hv0 = np.zeros([1, 28, 28, 1])
        inputs_shift_v0[:, :, :shift0[0], :] = image[:, (28-shift0[0]):, :].copy()
        inputs_shift_v0[:, :, shift0[0]:, :] = image[:, :(28-shift0[0]), :].copy()
        inputs_shift_hv0[:, :shift0[1], :, :] = inputs_shift_v0[:, (28-shift0[1]):, :, :]
        inputs_shift_hv0[:, shift0[1]:, :, :] = inputs_shift_v0[:, :(28-shift0[1]), :, :]
        shift1 = np.random.randint(7, size=2)
        inputs_shift_v1 = np.zeros([1, 28, 28, 1])
        inputs_shift_hv1 = np.zeros([1, 28, 28, 1])
        inputs_shift_v1[:, :, :shift1[0], :] = image[:, (28-shift1[0]):, :].copy()
        inputs_shift_v1[:, :, shift1[0]:, :] = image[:, :(28-shift1[0]), :].copy()
        inputs_shift_hv1[:, :shift1[1], :, :] = inputs_shift_v1[:, (28-shift1[1]):, :, :]
        inputs_shift_hv1[:, shift1[1]:, :, :] = inputs_shift_v1[:, :(28-shift1[1]), :, :]
        fig, axes = plt.subplots(2, 1)
        cax1 = axes[0].imshow(np.squeeze(inputs_shift_hv0), cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"First shift={shift0}")
        cax2 = axes[1].imshow(np.squeeze(inputs_shift_hv1), cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Second shift={shift1}")
        fig.colorbar(cax2, ax=axes)


def evaluate_shift_consistency_accuracy(models,model_names, x_test,y_test,max_shift=10):
    """ Evaluate both shift consistency and shift accuracy of the models
    we can use predict directly but predict_on_batch is much faster, except it requires more memory"""
    model_consistency_dict = {}
    model_accuracy_dict = {}
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
        model_accuracy = np.sum(predicted0 == np.argmax(y_test, axis=1) ) / len(y_test)
        model_accuracy_dict[model_name] = model_accuracy
        model_consistency = np.sum(predicted0 == predicted1) / len(x_test)
        model_consistency_dict[model_name] = model_consistency

    return model_consistency_dict, model_accuracy_dict


