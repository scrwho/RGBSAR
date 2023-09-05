##!/usr/bin/env python
# coding: utf-8


import pickle
from ModelsListDiffFuntions import *
import ModelsListDiffFuntions
# import nbimporter
# import import_ipynb


import os
import sys


def add_path_to_sys(path):
    module_path = os.path.abspath(path)
    if module_path not in sys.path:
        sys.path.append(module_path)


usePath = os.path.join(r'c:', os.sep, 'Users', 'scrwh',
                       'Documents', 'PythonScripts')
add_path_to_sys(usePath)


# List all the functions defined
# print(dir(ModelsListDiffFuntions))


FolderTree('Models')


# Locad the model
model_folder = os.path.join('Models/TrainingData1_2023-08-07 0040')
all_modelsb = get_file_list(model_folder)
all_modelsb


filtered_list = [file for file in all_modelsb if 'history' in file]
filtered_list


for f in filtered_list:
    n = os.path.splitext(os.path.basename(f))[0]
    print(n)


def load_models(filtered_list):
    model_list = []

    for file_path in filtered_list:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            model_list.append(model)

    return model_list


def get_model_names(filtered_list):
    model_names = []

    for f in filtered_list:
        name = os.path.basename(f)
        file_type = os.path.splitext(name)[1]
        name, _ = os.path.splitext(name)

        date_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{4})', name)
        if date_match:
            date = date_match.group(1)
            name = name.replace(f'_{date}', '')

        name = name.replace('wind_turbine_', '').replace(
            '_history_', '').replace('_history1', '').replace('-', '_')
        model_names.append(name)

    return model_names


def plot_training_histories(histories, model_names, output='results'):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    titles = ['Train Accuracy', 'Validation Accuracy',
              'Train Loss', 'Validation Loss']
    ylabels = ['Accuracy', 'Accuracy', 'Loss', 'Loss']
    legends = model_names

    for i, metric in enumerate(metrics):
        axs[i].set_title(titles[i])
        axs[i].set_ylabel(ylabels[i])
        axs[i].set_xlabel('Epoch')

        for j, history in enumerate(histories):
            if 'history' in history:
                history = history['history']
            axs[i].plot(history[metric])

        # Remove horizontal margins to make full use of the plot
        axs[i].margins(x=0)

        # Get the maximum number of epochs from all histories
        max_epochs = max([len(history[metric]) for history in histories])

        # Calculate the number of intervals on the x-axis
        num_intervals = 5  # Adjust this value to control the number of intervals
        interval = max_epochs // num_intervals

        # Set the x-axis tick positions and labels
        x_ticks = range(0, max_epochs + 1, interval)
        x_labels = [str(x) for x in x_ticks]
        axs[i].set_xticks(x_ticks)
        axs[i].set_xticklabels(x_labels)

        # Set the last epoch number as the x-axis label
        # axs[i].set_xlabel('Epoch (Last: {})'.format(max_epochs))

        axs[i].set_xlabel('Epoch')

    # Create a single legend outside the subplots with additional spacing and a title
    fig.legend(legends, loc='upper right',
               bbox_to_anchor=(1.16, 0.95), title='Legend')

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'training_histories.png'),
                bbox_inches='tight')
    plt.show()


get_model_names(filtered_list)


model_list = load_models(filtered_list)
model_names = get_model_names(filtered_list)
# model_list


plot_training_histories(model_list, model_names)
