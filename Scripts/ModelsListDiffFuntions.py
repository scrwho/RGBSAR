## !/usr/bin/env python
#coding: utf-8

# # Functions

# ## Libraries




from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing import image  # , ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img, plot_model  # , to_categorical
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from keras_sequential_ascii import keras2ascii
import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.applications import InceptionV3, ResNet50, ResNet101, InceptionResNetV2, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import optimizers, regularizers, mixed_precision, backend as K
from tensorflow.keras.layers import *
import tensorflow as tf
from pycore.tikzeng import *
from pycore.blocks import *
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import graphviz
import netCDF4 as nc
import netCDF4
import platform
import pynvml
import psutil
import GPUtil
from itertools import combinations
import itertools
import webcolors
import colorsys
import random
import math
import visualkeras
import tensorboard
import pytest
import os
import gc
import re
import sys
import time
import pickle
import subprocess
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from packaging import version
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFont
from IPython.display import display
import matplotlib.patches as mpatches
plt.ion()


# from tensorflow.keras import backend as K


# sys.path.append('../')




# get_system_memory, get_intel_gpu_memory, get_tpu_memory are missing from any of the categories

# ## File I/O and OS
# - file_count - Counts files in a folder
# - get_file_list - Gets file paths recursively
# - get_dir_list - Gets directory paths recursively
# - FolderTree - Prints folder structure with file counts
# - FileTree - Prints folder and file structure
# - get_image_paths - Gets image paths from input
# - CreateDir - Create a directory if it does not exists
# - get_model_names - Extract the names of Models from list of saved models
#




def CreateDir(dir):
    return os.makedirs(dir, exist_ok=True)


def file_count(folder):
    count = 0
    for root, dirs, files in os.walk(folder):
        count += len([f for f in files if os.path.isfile(os.path.join(root, f))])
    return count


def get_file_list(data_folder):
    return [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(data_folder) for file in files]


def get_dir_list(data_folder):
    return [os.path.join(dirpath, dirname) for dirpath, dirnames, _ in os.walk(data_folder) for dirname in dirnames]


def FolderTree(FolderPath):
    for root, dirs, files in os.walk(FolderPath):
        level = root.count(os.sep) - FolderPath.count(os.sep)
        indent = ' ' * 4 * level
        num_files = len(files)
        print(
            f"{indent}{os.path.basename(root)}/ ({num_files} file{'s' if num_files != 1 else ''})")


def FileTree(FolderPath):
    for root, dirs, files in os.walk(FolderPath):
        level = root.count(os.sep) - FolderPath.count(os.sep)
        indent = ' ' * 4 * level
        num_files = len(files)
        print(
            f"{indent}{os.path.basename(root)}/ ({num_files} file{'s' if num_files != 1 else ''})")
        for file in files:
            file_indent = ' ' * 4 * (level+1)
            print(f"{file_indent}{file}")


def get_image_paths(image_path):
    if isinstance(image_path, list):
        image_paths = [path for path in image_path if path.lower().endswith(
            ('.jpg', '.jpeg', '.png'))]
    elif os.path.isdir(image_path):
        image_paths = [os.path.join(root, file)
                       for root, dirs, files in os.walk(image_path)
                       for file in files
                       if file.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_paths = [image_path]
    return image_paths


def read_files(file_list):

    data = {}

    for f in file_list:
        # print(f)

        name = os.path.basename(f)
        file_type = os.path.splitext(name)[1]

        date_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{4})', name)
        if date_match:
            date = date_match.group(1)
            name = name.replace(f'_{date}', '')

        name = name.replace('wind_turbine_', '')
        name = name.replace(file_type, '')
        name = name.replace('-', '_')

        if f.endswith('.h5'):
            model = load_model(f)
            data[name] = model

        elif f.endswith('.csv'):
            df = pd.read_csv(f)
            data[name] = df

        elif f.endswith('.pkl'):
            with open(f, 'rb') as pkl:
                data[name] = pickle.load(pkl)

    return data


def read_filesPrint(file_list):

    # data = {}

    for f in file_list:
        # print(f)

        name = os.path.basename(f)
        file_type = os.path.splitext(name)[1]

        date_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{4})', name)
        if date_match:
            date = date_match.group(1)
            name = name.replace(f'_{date}', '')

        name = name.replace('wind_turbine_', '')
        name = name.replace(file_type, '')
        name = name.replace('-', '_')
        # print(f"{name} = {f}")

        if f.endswith('.h5'):
            print(
                f"{name} = load_model('{f}') \ntf.keras.backend.clear_session()  # Clear GPU memory \n")
            # model = load_model(f)
            # data[name] = model

        elif f.endswith('.csv'):
            print(f"{name}_df = pd.read_csv('{f}')")
            # df = pd.read_csv(f)
            # data[name] = df

        elif f.endswith('.pkl'):
            print(
                f"with open('{f}', 'rb') as pkl:\n    {name} = pickle.load(pkl)")
            # with open(f, 'rb') as pkl:
            #   data[name] = pickle.load(pkl)

    # return data


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


def CreateDir(dir):
    return os.makedirs(dir, exist_ok=True)


# ## System Info
#
#     get_system_memory - Gets system memory usage
#     get_intel_gpu_memory - Gets Intel GPU memory usage
#     get_tpu_memory - Gets TPU memory usage
#




def get_system_memory():
    svmem = psutil.virtual_memory()
    return {'memory_total': f"{svmem.total/(1024**3):.2f} GB", 'memory_used': f"{svmem.used/(1024**3):.2f} GB"}


def get_intel_gpu_memory():
    if platform.system() != 'Windows':
        return None
    meminfo = os.popen('wmic path win32_VideoController get AdapterRAM').read()
    return {'memory_total': f"{int(meminfo.split()[-1])/(1024**3):.2f} GB"}


def get_nvidia_gpu_memory():
    cmd = 'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    gpu_memory = [line.split(',') for line in output.split('\n')]
    return [{'id': i, 'memory_total': f"{int(mem[0])/1e3:.2f} GB", 'memory_used': f"{int(mem[1])/1e3:.2f} GB"} for i, mem in enumerate(gpu_memory)]


def get_tpu_memory():
    try:
        import tensorflow as tf
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='')
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        tpu_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
        # each TPU has 8GB memory
        return {'memory_total': f"{tpu_strategy.num_replicas_in_sync * 8:.2f} GB"}
    except:
        return None


def get_system_memory():

    memory = {}

    svmem = psutil.virtual_memory()
    memory['ram'] = {
        'total': f"{svmem.total/(1024**3):.2f} GB",
        'used': f"{svmem.used/(1024**3):.2f} GB"
    }

    if platform.system() == 'Windows':
        meminfo = os.popen(
            'wmic path win32_VideoController get AdapterRAM').read()
        memory['intel_gpu'] = {
            'total': f"{int(meminfo.split()[-1])/(1024**3):.2f} GB"
        }

    cmd = 'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    gpu_memory = [line.split(',') for line in output.split('\n')]
    memory['nvidia_gpu'] = [{
        'id': i,
        'total': f"{int(mem[0])/1e3:.2f} GB",
        'used': f"{int(mem[1])/1e3:.2f} GB"
    } for i, mem in enumerate(gpu_memory)]

    try:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='')
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        tpu_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
        memory['tpu'] = {
            'total': f"{tpu_strategy.num_replicas_in_sync * 8:.2f} GB"
        }
    except:
        pass

    return memory


# ## Data Sampling/Manipulation/Preprocessing
#
#
#     sample_list - Samples elements filtering by keyword
#     random_sample - Randomly samples elements
#     get_random_samples - Samples files from sub-folders
#     generate_data - Generates image data using ImageDataGenerator
#




def sample_list(trend, lists, size):
    data_list = [file for file in lists if trend in file]
    return random.sample(data_list, min(size, len(data_list)))


def random_sample(elements, n, seed=1234, replace=0):
    if seed is not None:
        random.seed(seed)
    if replace:
        return random.choices(elements, k=n)
    else:
        return random.sample(elements, k=n)

    import os


def get_random_samples(directory, num_samples, seed=1234, replace=0):
    """
    Get m random samples from n sub-directories of a directory, ensuring at least one sample is chosen from each sub-directory
    """
    if seed is not None:
        random.seed(seed)

    subdirectories = [subdir for subdir in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, subdir))]
    num_subdirectories = len(subdirectories)
    samples = []

    # Ensure at least one sample is chosen from each sub-directory
    for subdir in subdirectories:
        files = os.listdir(os.path.join(directory, subdir))
        if files:
            file = os.path.join(directory, subdir, random.choice(files))
            samples.append(file)

    # Remove duplicates from the list of samples
    samples = list(set(samples))

    # Choose remaining samples, allowing for replacement if necessary
    while len(samples) < num_samples:
        subdir = random.choice(subdirectories)
        files = os.listdir(os.path.join(directory, subdir))
        if files:
            file = os.path.join(directory, subdir, random.choice(files))
            if replace or file not in samples:
                samples = list(set(samples))
                samples.append(file)

    return samples





def generate_data(path, size, batch_size, class_mode, seed):
    data = ImageDataGenerator(rescale=1/255)
    return data.flow_from_directory(path,
                                    target_size=(size, size),
                                    batch_size=batch_size,
                                    class_mode=class_mode,
                                    seed=seed)


# ## Model Architecture
#
#     custom_model - Defines custom CNN model architecture
#     evaluate_models - Evaluates and compares multiple models
#

#
# ### Model 1
# ##### Note
# - data4 = Original sizes
# - data4b = 256
# - data4c = 512
# - data4d = 244
# -




def custom_model(data_dir, batch_size=32, epochs=10, img_size=244, num_classes=9):

    # Build the model architecture
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape=(
        img_size, img_size, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model




def evaluate_models(data_dir, batch_size=32, epochs=10, img_size=244):
    # Generate the Folder with the current date and time
    foldername = os.path.join(
        'Models', 'TrainingData1').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    Folder = f'{foldername}_{now}'
    os.makedirs(Folder, exist_ok=True)

    # Check if a GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("GPU available, training on GPU...")
        device_name = tf.test.gpu_device_name()
    else:
        print("GPU not available, training on CPU...")
        device_name = "/CPU:0"

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,  width_shift_range=0.2,
                                       height_shift_range=0.2,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Data augmentation for validation set
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training, validation and test set
    train_generator = load_data_generator(
        train_datagen, data_dir, 'train', img_size=img_size, batch_size=batch_size)
    val_generator = load_data_generator(
        val_datagen, data_dir, 'val', img_size=img_size, batch_size=batch_size)
    test_generator = load_data_generator(
        val_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    # Model creation
    models = [('Inception-V2', InceptionV3, 'imagenet'), ('ResNet-50', ResNet50, 'imagenet'),
              ('ResNet-101', ResNet101, 'imagenet'), ('Inception-ResNet-V2',
                                                      InceptionResNetV2, 'imagenet'),
              ('VGG', VGG16, 'imagenet'), ('Custom', custom_model, None)]
    models = [('Inception-V2', InceptionV3, 'imagenet'),
              ('Custom', custom_model, None)]

    # models = [('Inception-V2', InceptionV3, 'imagenet'),('Inception-ResNet-V2', InceptionResNetV2, 'imagenet'), ('Custom', custom_model, None)]

    model_metrics = {'Model': [], 'Training Accuracy': [],
                     'Validation Accuracy': [], 'Test Accuracy': []}

    for name, model_fn, weights in models:
        if model_fn == custom_model:
            model = model_fn(data_dir=data_dir, batch_size=batch_size,
                             epochs=epochs, img_size=img_size, num_classes=num_classes)
        else:
            base_model = model_fn(input_shape=(
                img_size, img_size, 3), include_top=False, weights=weights)
            x = Flatten()(base_model.output)
            x = Dense(units=512, activation='relu')(x)
            x = Dense(units=256, activation='relu')(x)
            output = Dense(train_generator.num_classes,
                           activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=output)

            for layer in base_model.layers:
                layer.trainable = False

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        # optimizer = Adam(learning_rate=0.001)
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Move the model to the GPU if available
        with tf.device(device_name):
            # Define the learning rate schedule
            def lr_schedule(epoch):
                learning_rate = 0.0001
                if epoch > 30:
                    learning_rate *= 0.1
                elif epoch > 20:
                    learning_rate *= 0.01
                print('Learning rate:', learning_rate)
                print(get_nvidia_gpu_memory())
                return learning_rate

            # Define the callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
            lr_scheduler = LearningRateScheduler(lr_schedule)
            checkpoint = ModelCheckpoint(
                f'{Folder}/best_{name}_model1.h5', monitor='val_loss', save_best_only=True)

            # Train the model
            print(f'Training Model: {name}')
            start_times = datetime.now()
            print(f'{name} Started: {start_times}')
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=len(val_generator),
                callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
            )
            end_times = datetime.now()
            print(f'{name} Ended: {end_times}')
            print(f'Duration: {end_times - start_times}')

            # Run the garbage collector
            gc.collect()

            now = datetime.now().strftime('%Y-%m-%d %H%M')
            # save your model and its history to disk
            model.save(f'{Folder}/wind_turbine_{name}_model1_{now}.h5')
            with open(f'{Folder}/wind_turbine_{name}_history1_{now}.pkl', 'wb') as f:
                pickle.dump(history.history, f)

        # Evaluation
        _, train_acc = model.evaluate(train_generator)
        _, val_acc = model.evaluate(val_generator)
        _, test_acc = model.evaluate(test_generator)
        model_metrics['Model'].append(name)
        model_metrics['Training Accuracy'].append(train_acc)
        model_metrics['Validation Accuracy'].append(val_acc)
        model_metrics['Test Accuracy'].append(test_acc)

    # Saving the performance in a DataFrame
    df = pd.DataFrame(model_metrics)
    df.set_index('Model', inplace=True)
    df['Average'] = df.select_dtypes(include='number').mean(axis=1)

    # Generate the filename with the current date and time
    name = os.path.join(Folder, 'TrainingData1').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    filename = f'{name}_{now}.csv'

    # Save the DataFrame to the CSV file
    df.to_csv(filename, index=True)

    print(df)

    # print(f'Saved DataFrame to {filename}')

    return df


# ### Models 1b
# - Use distributed training
# - multiple GPUs and want to take advantage of distributed training to potentially speed up the process



def evaluate_models1b(data_dir, batch_size=32, epochs=10, img_size=244):
    # Generate the Folder with the current date and time
    foldername = os.path.join(
        'Models', 'TrainingData1b').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    Folder = f'{foldername}_{now}'
    os.makedirs(Folder, exist_ok=True)

    # Determine device to use
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif len(devices) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,  width_shift_range=0.2,
                                       height_shift_range=0.2,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Data augmentation for validation set
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training, validation and test set
    train_generator = load_data_generator(
        train_datagen, data_dir, 'train', img_size=img_size, batch_size=batch_size)
    val_generator = load_data_generator(
        val_datagen, data_dir, 'val', img_size=img_size, batch_size=batch_size)
    test_generator = load_data_generator(
        val_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    # Model creation
    models = [('Inception-V2', InceptionV3, 'imagenet'), ('ResNet-50', ResNet50, 'imagenet'),
              ('ResNet-101', ResNet101, 'imagenet'), ('Inception-ResNet-V2',
                                                      InceptionResNetV2, 'imagenet'),
              ('VGG16', VGG16, 'imagenet'), ('Custom', custom_model, None)]
    # models = [('Inception-V2', InceptionV3, 'imagenet'),('Inception-ResNet-V2', InceptionResNetV2, 'imagenet'), ('Custom', custom_model, None)]

    model_metrics = {'Model': [], 'Training Accuracy': [],
                     'Validation Accuracy': [], 'Test Accuracy': []}

    for name, model_fn, weights in models:
        with strategy.scope():
            if model_fn == custom_model:
                model = model_fn(data_dir=data_dir, batch_size=batch_size,
                                 epochs=epochs, img_size=img_size, num_classes=num_classes)
            else:
                base_model = model_fn(input_shape=(
                    img_size, img_size, 3), include_top=False, weights=weights)
                x = Flatten()(base_model.output)
                x = Dense(units=512, activation='relu')(x)
                x = Dense(units=256, activation='relu')(x)
                output = Dense(train_generator.num_classes,
                               activation='softmax')(x)
                model = Model(inputs=base_model.input, outputs=output)

                for layer in base_model.layers:
                    layer.trainable = False

            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy', metrics=['accuracy'])

        # Define the learning rate schedule
        def lr_schedule(epoch):
            learning_rate = 0.0001
            if epoch > 30:
                learning_rate *= 0.1
            elif epoch > 20:
                learning_rate *= 0.01
            print('Learning rate:', learning_rate)
            print(get_nvidia_gpu_memory())
            return learning_rate

        # Define the callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        now = datetime.now().strftime('%Y-%m-%d %H%M')
        checkpoint = ModelCheckpoint(
            f'{Folder}/best_{name}_model1b_{now}.h5', monitor='val_loss', save_best_only=True)

        # Train the model
        print(f'Training Model: {name}')
        start_times = datetime.now()
        print(f'{name} Started: {start_times}')
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
        )
        end_times = datetime.now()
        print(f'{name} Ended: {end_times}')
        print(f'Duration: {end_times - start_times}')

        # Run the garbage collector
        gc.collect()

        now = datetime.now().strftime('%Y-%m-%d %H%M')
        # save your model and its history to disk
        model.save(f'{Folder}/wind_turbine_{name}_model1b_{now}.h5')
        with open(f'{Folder}/wind_turbine_{name}_history1b_{now}.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        # Evaluation
        _, train_acc = model.evaluate(train_generator)
        _, val_acc = model.evaluate(val_generator)
        _, test_acc = model.evaluate(test_generator)
        model_metrics['Model'].append(name)
        model_metrics['Training Accuracy'].append(train_acc)
        model_metrics['Validation Accuracy'].append(val_acc)
        model_metrics['Test Accuracy'].append(test_acc)

    # Saving the performance in a DataFrame
    df = pd.DataFrame(model_metrics)
    df.set_index('Model', inplace=True)
    df['Average'] = df.select_dtypes(include='number').mean(axis=1)

    # Generate the filename with the current date and time
    name = os.path.join(Folder, 'TrainingData1b').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    filename = f'{name}_{now}.csv'

    # Save the DataFrame to the CSV file
    df.to_csv(filename, index=True)

    print(df)

    # print(f'Saved DataFrame to {filename}')
    return df


#
# ### Model 2



def custom_modelB(data_dir, batch_size=32, epochs=10, img_size=244, num_classes=9):

    # Build the model architecture
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape=(
        img_size, img_size, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model




def evaluate_modelsb(data_dir, batch_size=32, epochs=10, img_size=244):
    # Generate the Folder with the current date and time
    foldername = os.path.join(
        'Models', 'TrainingData2').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    Folder = f'{foldername}_{now}'
    os.makedirs(Folder, exist_ok=True)

    # Check if a GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("GPU available, training on GPU...")
        device_name = tf.test.gpu_device_name()
    else:
        print("GPU not available, training on CPU...")
        device_name = "/CPU:0"

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,  width_shift_range=0.2,
                                       height_shift_range=0.2,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Data augmentation for validation set
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training, validation and test set
    train_generator = load_data_generator(
        train_datagen, data_dir, 'train', img_size=img_size, batch_size=batch_size)
    val_generator = load_data_generator(
        val_datagen, data_dir, 'val', img_size=img_size, batch_size=batch_size)
    test_generator = load_data_generator(
        val_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    # Model creation
    models = [('Inception-V2', InceptionV3, 'imagenet'), ('ResNet-50', ResNet50, 'imagenet'),
              ('ResNet-101', ResNet101, 'imagenet'), ('Inception-ResNet-V2',
                                                      InceptionResNetV2, 'imagenet'),
              ('VGG', VGG16, 'imagenet'), ('Custom', custom_model, None)]
    models = [('Inception-V2', InceptionV3, 'imagenet'),
              ('Custom', custom_model, None)]

    # models = [('Inception-V2', InceptionV3, 'imagenet'),('Inception-ResNet-V2', InceptionResNetV2, 'imagenet'), ('Custom', custom_model, None)]

    model_metrics = {'Model': [], 'Training Accuracy': [],
                     'Validation Accuracy': [], 'Test Accuracy': []}

    for name, model_fn, weights in models:
        if model_fn == custom_model:
            model = model_fn(data_dir=data_dir, batch_size=batch_size,
                             epochs=epochs, img_size=img_size, num_classes=num_classes)
        else:
            base_model = model_fn(input_shape=(
                img_size, img_size, 3), include_top=False, weights=weights)
            x = Flatten()(base_model.output)
            x = Dense(units=1024, activation='relu')(x)
            x = Dense(units=512, activation='relu')(x)
            output = Dense(train_generator.num_classes,
                           activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=output)

            for layer in base_model.layers:
                layer.trainable = False

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Move the model to the GPU if available
        with tf.device(device_name):
            # Define the learning rate schedule
            def lr_schedule(epoch):
                learning_rate = 0.0001
                if epoch > 30:
                    learning_rate *= 0.1
                elif epoch > 20:
                    learning_rate *= 0.01
                print('Learning rate:', learning_rate)
                print(get_nvidia_gpu_memory())
                return learning_rate

            # Define the callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
            lr_scheduler = LearningRateScheduler(lr_schedule)
            checkpoint = ModelCheckpoint(
                f'{Folder}/best_{name}_model2.h5', monitor='val_loss', save_best_only=True)

            # Train the model
            # print('Training Model:',name)
            print(f'Training Model: {name}')
            start_times = datetime.now()
            print(f'{name} Started: {start_times}')
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=len(val_generator),
                callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
            )
            end_times = datetime.now()
            print(f'{name} Ended: {end_times}')
            print(f'Duration: {end_times - start_times}')

            # Run the garbage collector
            gc.collect()

            now = datetime.now().strftime('%Y-%m-%d %H%M')
            # save your model and its history to disk
            model.save(f'{Folder}/wind_turbine_{name}_model2_{now}.h5')
            with open(f'{Folder}/wind_turbine_{name}_history2_{now}.pkl', 'wb') as f:
                pickle.dump(history.history, f)

        # labels = list(train_generator.class_indices.keys())
        # model.summary()

        # Evaluation
        _, train_acc = model.evaluate(train_generator)
        _, val_acc = model.evaluate(val_generator)
        _, test_acc = model.evaluate(test_generator)
        model_metrics['Model'].append(name)
        model_metrics['Training Accuracy'].append(train_acc)
        model_metrics['Validation Accuracy'].append(val_acc)
        model_metrics['Test Accuracy'].append(test_acc)

    # Saving the performance in a DataFrame
    df = pd.DataFrame(model_metrics)
    df.set_index('Model', inplace=True)
    df['Average'] = df.select_dtypes(include='number').mean(axis=1)

    # Generate the filename with the current date and time
    name = os.path.join(Folder, 'TrainingData2').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    filename = f'{name}_{now}.csv'

    # Save the DataFrame to the CSV file
    df.to_csv(filename, index=True)

    print(df)

    # print(f'Saved DataFrame to {filename}')

    return df


# ### Models 2b
# - Use distributed training
# - multiple GPUs and want to take advantage of distributed training to potentially speed up the process



def evaluate_modelsb2(data_dir, batch_size=32, epochs=10, img_size=244):
    # Generate the Folder with the current date and time
    foldername = os.path.join(
        'Models', 'TrainingData2b').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    Folder = f'{foldername}_{now}'
    os.makedirs(Folder, exist_ok=True)

    # Determine device to use
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif len(devices) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,  width_shift_range=0.2,
                                       height_shift_range=0.2,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Data augmentation for validation set
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training, validation and test set
    train_generator = load_data_generator(
        train_datagen, data_dir, 'train', img_size=img_size, batch_size=batch_size)
    val_generator = load_data_generator(
        val_datagen, data_dir, 'val', img_size=img_size, batch_size=batch_size)
    test_generator = load_data_generator(
        val_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    # Model creation
    models = [('Inception-V2', InceptionV3, 'imagenet'), ('ResNet-50', ResNet50, 'imagenet'),
              ('ResNet-101', ResNet101, 'imagenet'), ('Inception-ResNet-V2',
                                                      InceptionResNetV2, 'imagenet'),
              ('VGG16', VGG16, 'imagenet'), ('Custom', custom_modelB, None)]
    # models = [('Inception-V2', InceptionV3, 'imagenet'),('Inception-ResNet-V2', InceptionResNetV2, 'imagenet'), ('Custom', custom_model, None)]

    model_metrics = {'Model': [], 'Training Accuracy': [],
                     'Validation Accuracy': [], 'Test Accuracy': []}

    for name, model_fn, weights in models:
        with strategy.scope():
            if model_fn == custom_modelB:
                model = model_fn(data_dir=data_dir, batch_size=batch_size,
                                 epochs=epochs, img_size=img_size, num_classes=num_classes)
            else:
                base_model = model_fn(input_shape=(
                    img_size, img_size, 3), include_top=False, weights=weights)
                x = Flatten()(base_model.output)
                x = Dense(units=1024, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(x)
                x = Dropout(0.5)(x)
                x = Dense(units=512, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(x)
                x = Dropout(0.5)(x)
                output = Dense(units=train_generator.num_classes,
                               activation='softmax')(x)
                model = Model(inputs=base_model.input, outputs=output)

                for layer in base_model.layers:
                    layer.trainable = False

            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy', metrics=['accuracy'])

        # Define the learning rate schedule

        def lr_schedule(epoch):
            learning_rate = 0.0001
            if epoch > 30:
                learning_rate *= 0.1
            elif epoch > 20:
                learning_rate *= 0.01
            print('Learning rate:', learning_rate)
            print(get_nvidia_gpu_memory())
            return learning_rate

        # Define the callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        now = datetime.now().strftime('%Y-%m-%d %H%M')
        checkpoint = ModelCheckpoint(
            f'{Folder}/best_{name}_model2b_{now}.h5', monitor='val_loss', save_best_only=True)

        # Train the model
        # print('Training Model:',name)
        print(f'Training Model: {name}')
        start_times = datetime.now()
        print(f'{name} Started: {start_times}')
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
        )
        end_times = datetime.now()
        print(f'{name} Ended: {end_times}')
        print(f'Duration: {end_times - start_times}')

        # Run the garbage collector
        gc.collect()

        now = datetime.now().strftime('%Y-%m-%d %H%M')
        # save your model and its history to disk
        model.save(f'{Folder}/wind_turbine_{name}_model2b_{now}.h5')
        with open(f'{Folder}/wind_turbine_{name}_history2b_{now}.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        # labels = list(train_generator.class_indices.keys())
        # model.summary()

        # Evaluation
        _, train_acc = model.evaluate(train_generator)
        _, val_acc = model.evaluate(val_generator)
        _, test_acc = model.evaluate(test_generator)
        model_metrics['Model'].append(name)
        model_metrics['Training Accuracy'].append(train_acc)
        model_metrics['Validation Accuracy'].append(val_acc)
        model_metrics['Test Accuracy'].append(test_acc)

    # Saving the performance in a DataFrame
    df = pd.DataFrame(model_metrics)
    df.set_index('Model', inplace=True)
    df['Average'] = df.select_dtypes(include='number').mean(axis=1)

    # Generate the filename with the current date and time
    name = os.path.join(Folder, 'TrainingData2b').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    filename = f'{name}_{now}.csv'

    # Save the DataFrame to the CSV file
    df.to_csv(filename, index=True)

    print(df)

    # print(f'Saved DataFrame to {filename}')
    return df


# ### _

# ## Model Training
#
#     generate_training_params - Generates training parameters
#     load_data_generator - Loads data generator
#
# ##### Hyperparameter Tuning
#     calculate_steps_per_epoch_and_epochs - Calculates training parameters
#     calculate_batch_size_steps_per_epoch_and_epochs - Calculates training parameters


def generate_training_params(num_train, num_val, num_test):
    # Batch size
    batch_size = 32 if num_train <= 10000 else 38
    # batch_size = 32 if num_train <= 10000 else 64

    # Epochs
    epochs = 25 if num_train <= 10000 else 28
    # epochs = 25 if num_train <= 10000 else 50

    # Steps per epoch
    steps_per_epoch = num_train // batch_size

    # Validation steps
    validation_steps = num_val // batch_size

    return batch_size, epochs, steps_per_epoch, validation_steps


def load_data_generator(datagen, data_dir, subdir, img_size=244, batch_size=32):
    if subdir != 'test':
        shuffles = True
    else:
        shuffles = False
    return datagen.flow_from_directory(
        os.path.join(data_dir, subdir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffles
        # class_mode='categorical'
    )


def calculate_steps_per_epoch_and_epochs(num_samples, desired_epochs):
    batch_size = 128  # example batch size
    steps_per_epoch = num_samples // batch_size + 1
    epochs = desired_epochs
    while steps_per_epoch * epochs < num_samples:
        epochs += 1
    return steps_per_epoch, epochs


def calculate_batch_size_steps_per_epoch_and_epochs(sample, desired_epochs):
    num_samples = len(sample)
    batch_size = train_generator.batch_size
    steps_per_epoch = num_samples // batch_size + 1
    epochs = desired_epochs
    while steps_per_epoch * epochs < num_samples:
        epochs += 1
    return batch_size, steps_per_epoch, epochs


def calculate_batch_size_steps_per_epoch_and_epochsb(train_generator, desired_epochs):
    num_samples = train_generator.samples
    batch_size = train_generator.batch_size
    steps_per_epoch = num_samples // batch_size + 1
    epochs = desired_epochs
    while steps_per_epoch * epochs < num_samples:
        epochs += 1
    return batch_size, steps_per_epoch, epochs


# ## Model Evaluation
#
#     checkHistory - Checks if history object or dict
#     plot_training_history - Plots accuracy/loss history
#     plot_training_histories - plot accuracy/loss history for multiple models
#     EvaluateTest - Evaluates model on test set
#     calculate_metrics_multiclass -
#     detect_faults - Detects faults in images with predictions




def checkHistory(history):
    return history.history if 'history' in history else history


def plot_training_history(history, output='results'):

    CreateDir(output)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(os.path.join(output, 'training_history.png'),
                bbox_inches='tight')
    plt.show()


def plot_training_historyB(history, output='results'):

    CreateDir(output)
    plt.plot(history["accuracy"])
    plt.plot(history['val_accuracy'])
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])

    plt.savefig(os.path.join(output, 'training_historyB.png'),
                bbox_inches='tight')
    plt.show()





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
        name, _ = os.path.splitext(name)

        date_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{4})', name)
        if date_match:
            date = date_match.group(1)
            name = name.replace(f'_{date}', '')

        name = name.replace('wind_turbine_', '').replace(
            '_history_', '').replace('_history1', '').replace('_', '-')
        model_names.append(name)

    return model_names


def plot_training_histories(histories, model_names, output='results'):

    CreateDir(output)

    metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    titles = ['Train Accuracy', 'Validation Accuracy',
              'Train Loss', 'Validation Loss']
    ylabels = ['Accuracy', 'Accuracy', 'Loss', 'Loss']

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        axs[i].set_title(titles[i])
        axs[i].set_ylabel(ylabels[i])
        axs[i].set_xlabel('Epoch')

        for history in histories:
            if 'history' in history:
                history = history['history']
            axs[i].plot(history[metric])

        # Remove horizontal margins to make full use of the plot
        axs[i].margins(x=0)

        # Get the maximum number of epochs from all histories
        max_epochs = max(len(history[metric]) for history in histories)

        # Calculate the number of intervals on the x-axis
        num_intervals = 5  # Adjust this value to control the number of intervals
        interval = max_epochs // num_intervals

        # Set the x-axis tick positions and labels
        x_ticks = range(0, max_epochs + 1, interval)
        x_labels = [str(x) for x in x_ticks]
        axs[i].set_xticks(x_ticks)
        axs[i].set_xticklabels(x_labels)

        axs[i].set_xlabel('Epoch')

    # Create a single legend outside the subplots with additional spacing and a title
    fig.legend(model_names, loc='upper right',
               bbox_to_anchor=(1.16, 0.95), title='Legend')

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'training_histories.png'),
                bbox_inches='tight')
    plt.show()




def EvaluateTest(data_dir, model, history, batch_size=32, img_size=256, output='results'):

    CreateDir(output)

    # Data augmentation for test set
    test_datagen = ImageDataGenerator(rescale=1./255)
    # load_data_generator(datagen, data_dir, subdir, img_size = 244, batch_size = 32)
    test_generator = load_data_generator(
        test_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    # Create an empty dictionary to store time per image
    time_per_image = {}

    # Evaluate the model on the test set
    # print('\n Evaluate the model on the test set')
    start_time = time.time()
    scores = model.evaluate(
        test_generator, steps=len(test_generator), verbose=1)
    end_time = time.time()
    # print("Test loss:", scores[0])
    # print("Test accuracy:", scores[1])

    # Calculate time per image in seconds
    time_taken = end_time - start_time
    time_per_image['test'] = time_taken / len(test_generator.filenames)

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(
        time_per_image, orient='index', columns=['Time per image (s)'])
    print('EvaluateTime_report')
    print(df)
    df.to_csv(os.path.join(output, "EvaluateTime_report.csv"), index=True)

    history = checkHistory(history)

    # get the index of the highest accuracy value
    print('Training indexes')
    highest_acc_index = np.argmax(history['val_accuracy'])

    # create a dictionary with the values
    data = {'first': [history['loss'][0], history['accuracy'][0],
                      history['val_loss'][0], history['val_accuracy'][0]],
            'lowest': [min(history['loss']), min(history['accuracy']),
                       min(history['val_loss']), min(history['val_accuracy'])],
            'highest': [max(history['loss']), max(history['accuracy']),
                        max(history['val_loss']), max(history['val_accuracy'])],
            'last': [history['loss'][-1], history['accuracy'][-1],
                     history['val_loss'][-1], history['val_accuracy'][-1]]
            }

    # create a dataframe with the values as the index
    df = pd.DataFrame(data, index=[
                      'training loss', 'training accuracy', 'validation loss', 'validation accuracy'])
    # convert values to 4 decimal places
    df = df.round(4)
    # df = df.apply(lambda x: round(x, 4) if isinstance(x[1], float) else x)
    # save the dataframe as CSV
    # df.to_csv(os.path.join(output_dir, 'history.csv'), index=True)
    df.to_csv(os.path.join(output, 'history.csv'), index=True)
    # print the dataframe
    print(df, '\n')

    plot_training_history(history)

    plot_training_historyB(history)

    classNmaes = list(test_generator.class_indices.keys())

    # Evaluate the model on the test set
    print('\n Evaluate the model on the test set')
    # scores = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])

    # Get the true labels and the predicted labels from the test set
    true_labels = test_generator.classes
    predicted_labels = model.predict(
        test_generator, steps=len(test_generator), verbose=1)

    # Compute the binary true labels and predicted labels for each class
    class_names = list(test_generator.class_indices.keys())
    binary_true_labels = []
    binary_predicted_labels = []
    for i in range(len(class_names)):
        class_idx = test_generator.class_indices[class_names[i]]
        binary_true_labels.append((true_labels == class_idx).astype(int))
        binary_predicted_labels.append(predicted_labels[:, i])

    # Compute the ROC curve and AUC for each class
    plt.figure(figsize=(8, 6))
    auc_roc_scores = []
    for i in range(len(class_names)):
        fpr, tpr, thresholds = roc_curve(
            binary_true_labels[i], binary_predicted_labels[i])
        roc_auc = roc_auc_score(
            binary_true_labels[i], binary_predicted_labels[i])
        auc_roc_scores.append(roc_auc)
        plt.plot(fpr, tpr, lw=2,
                 label=class_names[i] + ' (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver operating characteristic ROC', fontsize=16)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output, 'auc_roc_scores.png'),
                bbox_inches='tight')
    plt.show()

    print("clases: \n", class_names)
    # Compute the classification report
    predicted_labels = np.argmax(predicted_labels, axis=1)
    cm = confusion_matrix(true_labels, predicted_labels)

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm = df_cm.round(2)
    df_cm.to_csv(os.path.join(output, 'confusion_matrix.csv'), index=True)

    cr = classification_report(
        true_labels, predicted_labels, target_names=class_names)
    cr_dict = classification_report(
        true_labels, predicted_labels, target_names=class_names, output_dict=True)
    df_cr = pd.DataFrame(cr_dict).transpose()
    df_cr = df_cr.round(2)
    df_cr.to_csv(os.path.join(output, "classification_report.csv"), index=True)

    print(" \nConfusion matrix:\n", cm)
    print(df_cm)

    # plot_confusion_matrix(cm, class_names)
    plot_confusion_matrixN(cm, class_names, normalize=True)

    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    plt.ylabel('Ground Truth label')
    name = "Confusion_matrix_normalized.png"
    plt.savefig(os.path.join(output, name))
    # plt.show()

    print(" \nClassification report:\n", cr)
    print(df_cr)

    # Create a table of AUC-ROC scores for each class
    auc_roc_table = pd.DataFrame(
        {'Class': class_names, 'AUC-ROC': auc_roc_scores})
    auc_roc_table = auc_roc_table.round(2)
    auc_roc_table.to_csv(os.path.join(
        output, "auc_roc_table.csv"), index=False)
    print("\nauc_roc_table:\n", auc_roc_table)

    lb = LabelBinarizer()
    true_labels_binary = lb.fit_transform(true_labels)
    predicted_labels_binary = lb.transform(predicted_labels)

    avg_precision = average_precision_score(
        true_labels_binary, predicted_labels_binary, average='macro')
    print("Average Precision Score: ", avg_precision)




def calculate_metrics_multiclass(y_true, y_pred, class_names):
    metrics = {}

    for i, class_name in enumerate(class_names):
        # Create binary label arrays
        y_true_class = (y_true == i).astype(int)
        y_pred_class = (y_pred == i).astype(int)

        # Calculate TP, FP, TN, FN
        tp = np.sum(y_true_class * y_pred_class)
        fp = np.sum((1 - y_true_class) * y_pred_class)
        tn = np.sum((1 - y_true_class) * (1 - y_pred_class))
        fn = np.sum(y_true_class * (1 - y_pred_class))

        # Calculate TPR and FPR
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        # Save metrics for the class
        metrics[class_name] = {'TP': tp, 'FP': fp,
                               'TN': tn, 'FN': fn, 'TPR': tpr, 'FPR': fpr}

    # Convert metrics to DataFrame
    df = pd.DataFrame(metrics)
    df = df.T
    df.index.name = 'Class'

    return df


# ### Model Inference/Prediction
#
#     detect_faults - Detects faults in images
#     detect_faultsdf - Detects faults and outputs dataframe
#     detect_faultsdfTrue - Detects faults without visualization
#     detect_faultsdfalone - Detects faults without grid visualization
#




def detect_faults(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)

    Returns:
    - predictions: List of tuples representing the fault predictions
        (fault_name, x1, y1, x2, y2, confidence)
    """

    CreateDir(output)

    if isinstance(image_path, list):
        image_paths = image_path
    elif os.path.isdir(image_path):
        image_paths = [os.path.join(root, file)
                       for root, dirs, files in os.walk(image_path)
                       for file in files
                       if file.endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_paths = [image_path]

    all_predictions = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        predictions = model.predict(image)

        colors = generate_colors(len(class_names))

        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(image))

        for i, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                x1, y1, x2, y2 = (10 * i, 10 * i, 200 - 10 * i, 200 - 10 * i)
                colord = colors[i]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color=colord)
                ax.add_patch(rect)
                ax.text(x1, y1, f'{class_names[i]} ({prediction:.2f})',
                        fontsize=14, fontweight='bold', color=colord)

        ax.set_title(path.split('\\')[-2])
        ax.axis('off')
        plt.show()

        predictions = [(class_names[i], x1, y1, x2, y2, prediction)
                       for i, prediction in enumerate(predictions[0])
                       if prediction > 0.5]
        all_predictions.append(predictions)

    return all_predictions





def detect_faultsdf(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)

    Returns:
    - predictions: List of tuples representing the fault predictions
        (fault_name, x1, y1, x2, y2, confidence)
    """

    CreateDir(output)

    if isinstance(image_path, list):
        image_paths = image_path
    elif os.path.isdir(image_path):
        image_paths = [os.path.join(root, file)
                       for root, dirs, files in os.walk(image_path)
                       for file in files
                       if file.endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_paths = [image_path]

    rows = 3 if len(image_paths) < 7 else 4
    numm = math.ceil(len(image_paths) / rows)
    # dim = (256, 256)

    plt.figure(figsize=(20, 15))
    gs1 = gridspec.GridSpec(numm, rows)
    gs1.update(wspace=0.025, hspace=0.08)

    all_predictions = []
    for idx, path in enumerate(image_paths):
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        predictions = model.predict(image)
        pred_labels = []
        confidences = []

        colors = generate_colors(len(class_names))

        ax1 = plt.subplot(gs1[idx])
        ax1.imshow(np.squeeze(image))

        for i, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                pred_labels.append(class_names[i])
                confidences.append(prediction)

                x1, y1, x2, y2 = (10 * i, 10 * i, 200 - 10 * i, 200 - 10 * i)
                colord = colors[i]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color=colord)
                ax1.add_patch(rect)
                ax1.text(x1, y1, f'{class_names[i]} ({prediction:.2f})',
                         fontsize=14, fontweight='bold', color=colord)

        gt_label = path.split(os.path.sep)[-2]
        pred_label = ','.join(pred_labels) if pred_labels else 'None'
        confidence = ','.join(
            [f'{c:.2f}' for c in confidences]) if confidences else 'None'

        ax1.set_title(path.split(os.path.sep)[-2])
        ax1.axis('off')

        predictions = [(class_names[i], x1, y1, x2, y2, prediction)
                       for i, prediction in enumerate(predictions[0])
                       if prediction > 0.5]

        all_predictions.append(
            (os.path.basename(path), gt_label, pred_label, confidence))
        # all_predictions.append(predictions)

    plt.show()

    df = pd.DataFrame(all_predictions, columns=[
                      'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'])

    df_split = df.assign(Ground_Truth_Label=df['Ground Truth Label']).assign(
        Predicted_Label=df['Predicted Label'].str.split(',')).explode('Predicted_Label')

    count = len(df_split[df_split['Ground_Truth_Label']
                == df_split['Predicted_Label']])
    countb = len(df_split[df_split['Ground_Truth_Label']
                 != df_split['Predicted_Label']])
    print(f"Total predictions that match each Ground Truth: {count}")
    print(f"Total predictions that do not match each Ground Truth: {countb}")

    return df




def detect_faultsdf(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)
    - img_size: Image size to resize to (default 256)
    - threshold: Prediction threshold (default 0.2)
    - n_cols: Number of columns in the final image grid (default 3)

    Returns:
    - df: Pandas dataframe with columns 'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'
    """

    CreateDir(output)

    sepps = os.path.sep

    image_paths = get_image_paths(image_path)
    n_cols = 3 if len(image_paths) < 7 else 4

    all_predictions = []
    predicted_images = []
    n_rows = int(np.ceil(len(image_paths) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    colors = generate_colors(len(class_names))

    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        predictions = model.predict(image)
        pred_labels = []
        confidences = []

        ax = axes[i]
        ax.imshow(np.squeeze(image))

        for j, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                pred_labels.append(class_names[j])
                confidences.append(prediction)

                x1, y1, x2, y2 = (10 * j, 10 * j, 200 - 10 * j, 200 - 10 * j)
                colord = colors[j]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color=colord)
                ax.add_patch(rect)
                ax.text(x1, y1, f'{class_names[j]} ({prediction:.2f})',
                        fontsize=14, fontweight='bold', color=colord)

        gt_label = path.split(os.path.sep)[-2]
        pred_label = ','.join(pred_labels) if pred_labels else 'None'
        confidence = ','.join(
            [f'{c:.2f}' for c in confidences]) if confidences else 'None'

        ax.set_title(path.split(sepps)[-2])
        ax.axis('off')

        predicted_images.append(fig)

        all_predictions.append(
            (os.path.basename(path), gt_label, pred_label, confidence))

    df = pd.DataFrame(all_predictions, columns=[
                      'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'])

    for i in range(len(image_paths), n_rows * n_cols):
        fig.delaxes(axes[i])

    # modify the spacing between subplots
    fig.subplots_adjust(wspace=0.005, hspace=0.15)
    # plt.tight_layout()
    plt.savefig(os.path.join(output, 'final_image.png'),
                bbox_inches='tight', pad_inches=0)
    # plt.show()

    df_split = df.assign(Ground_Truth_Label=df['Ground Truth Label']).assign(
        Predicted_Label=df['Predicted Label'].str.split(',')).explode('Predicted_Label')
    count = len(df_split[df_split['Ground_Truth_Label']
                == df_split['Predicted_Label']])
    countb = len(df_split[df_split['Ground_Truth_Label']
                 != df_split['Predicted_Label']])
    print(f"Total predictions that match each Ground Truth: {count}")
    print(f"Total predictions that do not match each Ground Truth: {countb}")

    # Save the DataFrame to the CSV file
    df.to_csv(os.path.join(output, 'SamplePredicted.csv'), index=False)

    return df




def detect_faultsdfTrue(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)
    - img_size: Image size to resize to (default 256)
    - threshold: Prediction threshold (default 0.2)

    Returns:
    - df: Pandas dataframe with columns 'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'
    """

    sepps = os.path.sep

    image_paths = get_image_paths(image_path)

    all_predictions = []

    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        predictions = model.predict(image)
        pred_labels = []
        confidences = []

        for j, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                pred_labels.append(class_names[j])
                confidences.append(prediction)

        gt_label = path.split(os.path.sep)[-2]
        pred_label = ','.join(pred_labels) if pred_labels else 'None'
        confidence = ','.join(
            [f'{c:.2f}' for c in confidences]) if confidences else 'None'

        all_predictions.append(
            (os.path.basename(path), gt_label, pred_label, confidence))

    df = pd.DataFrame(all_predictions, columns=[
                      'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'])

    df_split = df.assign(Ground_Truth_Label=df['Ground Truth Label']).assign(
        Predicted_Label=df['Predicted Label'].str.split(',')).explode('Predicted_Label')
    count = len(df_split[df_split['Ground_Truth_Label']
                == df_split['Predicted_Label']])
    countb = len(df_split[df_split['Ground_Truth_Label']
                 != df_split['Predicted_Label']])
    print(f"Total predictions that match each Ground Truth: {count}")
    print(f"Total predictions that do not match each Ground Truth: {countb}")

    # Save the DataFrame to the CSV file
    df.to_csv(os.path.join(output, 'SamplePredicted.csv'), index=False)

    return df




def detect_faultsdf(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)
    - img_size: Image size to resize to (default 256)
    - threshold: Prediction threshold (default 0.2)
    - n_cols: Number of columns in the final image grid (default 3)

    Returns:
    - df: Pandas dataframe with columns 'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'
    """

    CreateDir(output)

    sepps = os.path.sep
    pattern = r'[\[\](){}_\s]'

    image_paths = get_image_paths(image_path)
    n_cols = 3 if len(image_paths) < 7 else 4

    all_predictions = []
    predicted_images = []
    n_rows = int(np.ceil(len(image_paths) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    colors = generate_colors(len(class_names))

    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        start_time = time.time()
        predictions = model.predict(image)
        end_time = time.time()

        prediction_time = end_time - start_time
        prediction_time_in_sec = f"{prediction_time:.2f}"

        pred_labels = []
        confidences = []

        ax = axes[i]
        ax.imshow(np.squeeze(image))

        for j, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                pred_labels.append(class_names[j])
                confidences.append(prediction)

                x1, y1, x2, y2 = (10 * j, 10 * j, 200 - 10 * j, 200 - 10 * j)
                colord = colors[j]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color=colord)
                ax.add_patch(rect)
                ax.text(x1, y1, f'{class_names[j]} ({prediction:.2f})',
                        fontsize=14, fontweight='bold', color=colord)

        gt_label = path.split(os.path.sep)[-2]
        pred_label = ','.join(pred_labels) if pred_labels else 'None'
        confidence = ','.join(
            [f'{c:.2f}' for c in confidences]) if confidences else 'None'

        ax.set_title(path.split(sepps)[-2])
        ax.axis('off')

        predicted_images.append(fig)

        all_predictions.append((os.path.basename(
            path), gt_label, pred_label, confidence, prediction_time_in_sec))

    df = pd.DataFrame(all_predictions, columns=[
                      'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence', 'Prediction Time (sec)'])
    df['Image'] = df['Image'].str.replace(pattern, '')
    df['Image'] = df['Image'].str.replace('[\(\)\[\]\{\}]', '')
    df['Image'] = df['Image'].str[-9:]
    df['Image'] = df['Image'].str.capitalize()

    for i in range(len(image_paths), n_rows * n_cols):
        fig.delaxes(axes[i])

    # modify the spacing between subplots
    fig.subplots_adjust(wspace=0.005, hspace=0.15)
    plt.savefig(os.path.join(output, 'final_image.png'),
                bbox_inches='tight', pad_inches=0)

    df_split = df.assign(Ground_Truth_Label=df['Ground Truth Label']).assign(
        Predicted_Label=df['Predicted Label'].str.split(',')).explode('Predicted_Label')
    count = len(df_split[df_split['Ground_Truth_Label']
                == df_split['Predicted_Label']])
    countb = len(df_split[df_split['Ground_Truth_Label']
                 != df_split['Predicted_Label']])
    # count number of rows with one predicted label
    one_label = (df['Predicted Label'].str.count(',') == 0).sum()
    # count number of rows with more than one predicted label
    more_labels = (df['Predicted Label'].str.count(',') > 0).sum()

    total_Images = df.shape[0]
    total_Predictions = df_split.shape[0]
    # Assuming the variables are already defined
    print(f"\n\n\nNumber of rows with one predicted label: {one_label}")
    print(f"Number of rows with more than one predicted label: {more_labels}")
    print(f"Total Images label: {total_Images}")
    print(f"Total predicted label: {total_Predictions}")

    print(f"Total predictions that match each Ground Truth: {count}")
    print(f"Total predictions that do not match each Ground Truth: {countb}")

    # Calculate percentage of accuracy and wrong predictions
    accuracy = (count / total_Predictions) * 100
    wrong_predictions = (countb / total_Predictions) * 100

    print(f"Percentage of accuracy: {accuracy:.2f}%")
    print(f"Percentage of wrong predictions: {wrong_predictions:.2f}%\n\n")

    # Save the DataFrame to the CSV file
    df.to_csv(os.path.join(output, 'SamplePredicted.csv'), index=False)

    return df




def detect_faultsdfalone(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)
    - img_size: Image size to resize to (default 256)
    - threshold: Prediction threshold (default 0.2)

    Returns:
    - df: Pandas dataframe with columns 'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'
    """

    CreateDir(output)

    sepps = os.path.sep

    image_paths = get_image_paths(image_path)

    all_predictions = []

    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        start_time = time.time()
        predictions = model.predict(image)
        end_time = time.time()

        prediction_time = end_time - start_time
        prediction_time_in_sec = f"{prediction_time:.2f}"

        pred_labels = []
        confidences = []

        for j, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                pred_labels.append(class_names[j])
                confidences.append(prediction)

        gt_label = path.split(os.path.sep)[-2]
        pred_label = ','.join(pred_labels) if pred_labels else 'None'
        confidence = ','.join(
            [f'{c:.2f}' for c in confidences]) if confidences else 'None'

        all_predictions.append((os.path.basename(
            path), gt_label, pred_label, confidence, prediction_time_in_sec))

    # df = pd.DataFrame(all_predictions, columns=['Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'])
    df = pd.DataFrame(all_predictions, columns=[
                      'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence', 'Prediction Time (sec)'])

    df_split = df.assign(Ground_Truth_Label=df['Ground Truth Label']).assign(
        Predicted_Label=df['Predicted Label'].str.split(',')).explode('Predicted_Label')
    count = len(df_split[df_split['Ground_Truth_Label']
                == df_split['Predicted_Label']])
    countb = len(df_split[df_split['Ground_Truth_Label']
                 != df_split['Predicted_Label']])
    # count number of rows with one predicted label
    one_label = (df['Predicted Label'].str.count(',') == 0).sum()
    # count number of rows with more than one predicted label
    more_labels = (df['Predicted Label'].str.count(',') > 0).sum()

    total_Images = df.shape[0]
    total_Predictions = df_split.shape[0]
    # Assuming the variables are already defined
    print(f"\n\n\nNumber of rows with one predicted label: {one_label}")
    print(f"Number of rows with more than one predicted label: {more_labels}")
    print(f"Total Images label: {total_Images}")
    print(f"Total predicted label: {total_Predictions}")

    print(f"Total predictions that match each Ground Truth: {count}")
    print(f"Total predictions that do not match each Ground Truth: {countb}")

    # Calculate percentage of accuracy and wrong predictions
    accuracy = (count / total_Predictions) * 100
    wrong_predictions = (countb / total_Predictions) * 100

    print(f"Percentage of accuracy: {accuracy:.2f}%")
    print(f"Percentage of wrong predictions: {wrong_predictions:.2f}%\n\n")

    # Save the DataFrame to the CSV file
    df.to_csv(os.path.join(output, 'SamplePredicted.csv'), index=False)

    return df




def detect_faultsdfaloneb(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):

    CreateDir(output)
    sepps = os.path.sep

    image_paths = get_image_paths(image_path)

    all_predictions = []

    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        start_time = time.time()
        predictions = model.predict(image)
        end_time = time.time()

        prediction_time = end_time - start_time
        prediction_time_in_sec = f"{prediction_time:.2f}"

        pred_labels = []
        confidences = []

        for j, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                pred_labels.append(class_names[j])
                confidences.append(prediction)

        gt_label = path.split(os.path.sep)[-2]
        pred_label = ','.join(pred_labels) if pred_labels else 'None'
        confidence = ','.join(
            [f'{c:.2f}' for c in confidences]) if confidences else 'None'

        all_predictions.append((os.path.basename(
            path), gt_label, pred_label, confidence, prediction_time_in_sec))

    # df = pd.DataFrame(all_predictions, columns=['Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'])
    df = pd.DataFrame(all_predictions, columns=[
                      'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence', 'Prediction Time (sec)'])

    df_split = df.assign(Ground_Truth_Label=df['Ground Truth Label']).assign(
        Predicted_Label=df['Predicted Label'].str.split(',')).explode('Predicted_Label')

    # Calculate TP, FP, TN, FN for each image
    for image_name, df_image in df_split.groupby('Image'):
        gt_labels = set(df_image['Ground_Truth_Label'].values)
        pred_labels = set(df_image['Predicted_Label'].values)
        TP = len(gt_labels.intersection(pred_labels))
        FP = len(pred_labels - gt_labels)
        TN = 0
        FN = len(gt_labels - pred_labels)
        df.loc[df['Image'] == image_name, 'TP'] = TP
        df.loc[df['Image'] == image_name, 'FP'] = FP
        df.loc[df['Image'] == image_name, 'TN'] = TN
        df.loc[df['Image'] == image_name, 'FN'] = FN

    # Calculate TPR and FPR
    TP = df['TP'].sum()
    FP = df['FP'].sum()
    TN = df['TN'].sum()
    FN = df['FN'].sum()
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    print(f"TPR: {TPR:.2f}")
    print(f"FPR: {FPR:.2f}")

    # Save the DataFrame to the CSV file
    df.to_csv(os.path.join(output, 'SamplePredictedb.csv'), index=False)

    return df




def detect_faults_images(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):

    CreateDir(output)
    sepps = os.path.sep

    image_paths = get_image_paths(image_path)
    n_cols = min(4, len(image_paths))

    all_predictions = []
    data = []

    fig, axes = plt.subplots(len(image_paths), n_cols,
                             figsize=(4 * n_cols, 4 * len(image_paths)))
    axes = axes.flatten()

    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        predictions = model.predict(image)

        pred_labels = []
        confidences = []
        colors = generate_colors(len(class_names))

        ax = axes[i*n_cols:(i+1)*n_cols]
        ax[0].imshow(np.squeeze(image))
        ax[0].set_title(path.split(sepps)[-2])
        ax[0].axis('off')

        for j, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                pred_labels.append(class_names[j])
                confidences.append(prediction)

                x1, y1, x2, y2 = (10 * j, 10 * j, 200 - 10 * j, 200 - 10 * j)
                colord = colors[j]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color=colord)
                ax[0].add_patch(rect)
                ax[0].text(x1, y1, f'{class_names[j]} ({prediction:.2f})',
                           fontsize=14, fontweight='bold', color=colord)

        gt_label = path.split(sepps)[-2]
        pred_label = ','.join(pred_labels) or 'None'
        confidence = ','.join(f'{c:.2f}' for c in confidences) or 'None'

        all_predictions.append(
            (os.path.basename(path), gt_label, pred_label, confidence))

        for j, prediction in enumerate(predictions[0]):
            if prediction > threshold:
                ax[j].imshow(np.squeeze(image))
                ax[j].set_title(f'{class_names[j]} ({prediction:.2f})')
                ax[j].axis('off')

                predictions = [(class_names[j], x1, y1, x2, y2, prediction)
                               for j, prediction in enumerate(predictions[0])
                               if prediction > threshold]
                for prediction in predictions:
                    data.append([os.path.basename(path), gt_label,
                                prediction[0], prediction[5]])

        for k in range(j+1, n_cols):
            ax[k].axis('off')

    df = pd.DataFrame(all_predictions, columns=[
                      'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'])

    fig.subplots_adjust(wspace=0.05, hspace=0.005)
    # plt.show()
    plt.savefig(os.path.join(output, 'final_image.png'),
                bbox_inches='tight', pad_inches=0)

    df_split = df.assign(Predicted_Label=df['Predicted Label'].str.split(
        ',')).explode('Predicted_Label')
    count = (df_split['Ground Truth Label'] ==
             df_split['Predicted_Label']).sum()
    countb = (df_split['Ground Truth Label'] !=
              df_split['Predicted_Label']).sum()
    print(f"Total predictions that match each Ground Truth: {count}")
    print(f"Total predictions that do not match each Ground Truth: {countb}")

    return df




def detect_faults1(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)

    Returns:
    - predictions: Pandas dataframe with columns ['Image', 'Ground Truth Label', 'Predicted Label', 'Confidence']
    """

    CreateDir(output)

    if isinstance(image_path, list):
        image_paths = image_path
    elif os.path.isdir(image_path):
        image_paths = [os.path.join(root, file)
                       for root, dirs, files in os.walk(image_path)
                       for file in files
                       if file.endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_paths = [image_path]

    all_predictions = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        predictions = model.predict(image)

        pred_labels = []
        confidences = []
        for i, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                pred_labels.append(class_names[i])
                confidences.append(prediction)

        gt_label = path.split(os.path.sep)[-2]
        pred_label = ','.join(pred_labels) if pred_labels else 'None'
        confidence = ','.join(
            [f'{c:.2f}' for c in confidences]) if confidences else 'None'

        all_predictions.append(
            (os.path.basename(path), gt_label, pred_label, confidence))

    df = pd.DataFrame(all_predictions, columns=[
                      'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'])

    df_split = df.assign(Ground_Truth_Label=df['Ground Truth Label']).assign(
        Predicted_Label=df['Predicted Label'].str.split(',')).explode('Predicted_Label')

    count = len(df_split[df_split['Ground_Truth_Label']
                == df_split['Predicted_Label']])
    countb = len(df_split[df_split['Ground_Truth_Label']
                 != df_split['Predicted_Label']])
    print(f"Total predictions that match each Ground Truth: {count}")
    print(f"Total predictions that do not match each Ground Truth: {countb}")

    return df


def detect_faults2(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)

    Returns:
    - predictions: List of tuples representing the fault predictions
        (fault_name, x1, y1, x2, y2, confidence)
    """

    CreateDir(output)
    if isinstance(image_path, list):
        image_paths = image_path
    elif os.path.isdir(image_path):
        image_paths = [os.path.join(root, file)
                       for root, dirs, files in os.walk(image_path)
                       for file in files
                       if file.endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_paths = [image_path]

    all_predictions = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        predictions = model.predict(image)

        colors = generate_colors(len(class_names))

        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(image))

        predicted_labels = []
        for i, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                x1, y1, x2, y2 = (10 * i, 10 * i, 200 - 10 * i, 200 - 10 * i)
                colord = colors[i]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color=colord)
                ax.add_patch(rect)
                label = f'{class_names[i]} ({prediction:.2f})'
                ax.text(x1, y1, label,
                        fontsize=14, fontweight='bold', color=colord)
                predicted_labels.append((class_names[i], prediction))

        ax.set_title(path.split('\\')[-2])
        ax.axis('off')
        plt.show()

        true_label = path.split('\\')[-2]
        predictions_df = pd.DataFrame(predicted_labels, columns=[
                                      'Predicted Label', 'Confidence'])
        predictions_df['Image'] = os.path.basename(path)
        predictions_df['Ground Truth Label'] = true_label
        predictions_df = predictions_df[[
            'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence']]
        all_predictions.append(predictions_df)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    # print(all_predictions_df)
    return all_predictions_df


def detect_faults3(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)

    Returns:
    - predictions_df: pandas DataFrame representing the fault predictions for all images
    """

    CreateDir(output)

    if isinstance(image_path, list):
        image_paths = image_path
    elif os.path.isdir(image_path):
        image_paths = [os.path.join(root, file)
                       for root, dirs, files in os.walk(image_path)
                       for file in files
                       if file.endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_paths = [image_path]

    all_predictions = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        predictions = model.predict(image)

        colors = generate_colors(len(class_names))

        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(image))

        predicted_labels = []
        confidences = []
        for i, prediction in enumerate(predictions[0]):
            if prediction >= threshold:
                x1, y1, x2, y2 = (10 * i, 10 * i, 200 - 10 * i, 200 - 10 * i)
                colord = colors[i]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color=colord)
                ax.add_patch(rect)
                label = class_names[i]
                predicted_labels.append(label)
                confidences.append(prediction)
                ax.text(x1, y1, f'{label} ({prediction:.2f})',
                        fontsize=14, fontweight='bold', color=colord)

        ax.set_title(path.split('\\')[-2])
        ax.axis('off')
        plt.show()

        # create a list of dictionaries for each predicted label and confidence
        predictions_dict_list = []
        for label, confidence in zip(predicted_labels, confidences):
            predictions_dict_list.append({
                'Image': path.split('\\')[-1],
                'Ground Truth Label': path.split('\\')[-2],
                'Predicted Label': label,
                'Confidence': confidence
            })

        # create a DataFrame from the list of dictionaries
        predictions_df = pd.DataFrame(predictions_dict_list)

        all_predictions.append(predictions_df)

    # concatenate all the predictions DataFrames into one
    predictions_df = pd.concat(all_predictions)

    # reset the index of the DataFrame
    predictions_df = predictions_df.reset_index(drop=True)

    return predictions_df


def detect_faults4(model, image_path, class_names, img_size=256, threshold=0.2, output='results'):
    """
    Detects faults in a wind turbine blade image using a trained CNN model.

    Args:
    - model: Trained Keras model object
    - image_path: File path of the input image or directory
    - class_names: List of class names (in order of model output)
    - img_size: Image size (default=256)
    - threshold: Confidence threshold for predictions (default=0.2)

    Returns:
    - predictions: DataFrame representing the fault predictions
        (Image, Ground Truth Label, Predicted Label, Confidence)
    """

    CreateDir(output)

    if isinstance(image_path, list):
        image_paths = image_path
    elif os.path.isdir(image_path):
        image_paths = [os.path.join(root, file)
                       for root, dirs, files in os.walk(image_path)
                       for file in files
                       if file.endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_paths = [image_path]

    all_predictions = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (img_size, img_size))
        image = np.expand_dims(image, axis=0) / 255.

        predictions = model.predict(image)

        predicted_classes = [class_names[i]
                             for i, p in enumerate(predictions[0]) if p >= threshold]
        # predicted_confidences = [p for i, p in enumerate(predictions[0]) if p >= threshold]
        predicted_confidences = [round(p, 2) for i, p in enumerate(
            predictions[0]) if p >= threshold]

        true_class = path.split(os.path.sep)[-2]
        image_name = path.split(os.path.sep)[-1]

        predictions = [(image_name, true_class, predicted_class, confidence)
                       for predicted_class, confidence in zip(predicted_classes, predicted_confidences)]

        all_predictions.extend(predictions)

    df = pd.DataFrame(all_predictions, columns=[
                      'Image', 'Ground Truth Label', 'Predicted Label', 'Confidence'])

    # print(predictions_df)

    count = len(df[df['Ground Truth Label'] == df['Predicted Label']])
    count1 = len(df[df['Ground Truth Label'] != df['Predicted Label']])
    print('\nCorrect predictions:', count,
          '\nWrong predictions:', count1, '\n')

    return df


def ossep():
    return os.path.sep


# ## Visualization
#
#     generate_colors - Generates distinct colors
#     plot_image_grid - Plots a grid of images
#     plot_confusion_matrix - Plots confusion matrix
#     predictBar - Plots class probabilities as bars
#     predictBarColor - Plots class probabilities as colored bars
#     plot_probability_distribution_and_roc_curves_one_vs_rest - Plots probability distributions and ROC curves
#




def generate_colors(n, seed=None):
    if seed is not None:
        random.seed(seed)
    # Define major colors in order of perceived visibility
    major_colors = ["blue", "orange", "black",
                    "red", "green", "yellow", "purple", "pink"]
    # Convert major colors to HTML color codes and RGB values
    major_colors_hex_rgb = [(webcolors.name_to_hex(
        color), webcolors.name_to_rgb(color)) for color in major_colors]
    # Generate colors
    hues = [random.uniform(0, 1) for _ in range(n - len(major_colors))]
    saturations = [random.uniform(0.5, 1)
                   for _ in range(n - len(major_colors))]
    lightnesses = [random.uniform(0.4, 0.8)
                   for _ in range(n - len(major_colors))]
    hue_offsets = [random.choice([-0.2, 0, 0.2])
                   for _ in range(n - len(major_colors))]
    colors_hls = [(hues[i]+hue_offsets[i], saturations[i], lightnesses[i])
                  for i in range(n - len(major_colors))]
    colors_rgb = [tuple(round(c*255)
                        for c in colorsys.hls_to_rgb(*hls)) for hls in colors_hls]
    # Sort colors by proximity to the most distinguishable and visible colors
    colors_hex_rgb = sorted(major_colors_hex_rgb + list(zip(map(webcolors.rgb_to_hex, colors_rgb), colors_rgb)), key=lambda c: min(
        [abs(colorsys.rgb_to_hls(*c[1])[0] - colorsys.rgb_to_hls(*webcolors.name_to_rgb(color))[0]) for color in major_colors]))
    # Convert RGB values to HTML color codes and return the first n colors
    colors_hex = [c[0] for c in colors_hex_rgb]
    return colors_hex[:n]


def plot_image_grid(imglist):
    rows = 3 if len(imglist) < 7 else 4
    numm = math.ceil(len(imglist) / rows)
    dim = (256, 256)

    plt.figure(figsize=(20, 15))
    gs1 = gridspec.GridSpec(numm, rows)
    gs1.update(wspace=0.025, hspace=0.08)

    for idx, img_path in enumerate(imglist):
        shp = Image.open(img_path).size
        title = f'Image{idx} {shp}'
        img = Image.open(img_path).resize(dim)
        ax1 = plt.subplot(gs1[idx])
        plt.title(title)
        plt.grid(False)
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        plt.imshow(img)




# To get better visual of the confusion matrix:
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, output='results'):
    # Add Normalization Option
    '''prints pretty confusion metric with normalization option '''

    CreateDir(output)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# To get better visual of the confusion matrix:
def plot_confusion_matrixN(cm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues, output='results'):
    # Add Normalization Option
    '''prints pretty confusion metric with normalization option '''

    CreateDir(output)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        name = "Normalized confusion matrix.png"
        print("Normalized_confusion_matrix")
    else:
        print('Confusion matrix, without normalization')
        name = "Confusion_matrix_without_normalization.png"

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output, name))




def predictBar(model, image_path, classes, img_size=244, num_images=None, output='results'):
    """
    Predicts the class probabilities for a set of images using the specified model.

    Args:
        model (tensorflow.keras.Model): The model to use for prediction.
        image_path (str): The path to the directory/list containing the images to predict.
        classes (list of str): A list of the class names in the order predicted by the model.
        img_size (int, optional): The target size to resize the images. Defaults to 244.
        num_images (int, optional): The number of images to plot the predictions for.

    Returns:
        None
    """

    CreateDir(output)

    imglist = get_image_paths(image_path)
    num_images = num_images or len(imglist)
    images = [img_to_array(load_img(img_path, target_size=(
        img_size, img_size))) / 255.0 for img_path in imglist[:num_images]]

    # Predict the classes for the images
    predictions = model.predict(np.array(images))

    # Plot the images and predictions
    fig, axes = plt.subplots(num_images, 2, figsize=(16, 4 * num_images))
    fig.subplots_adjust(hspace=0.4, wspace=-0.2)

    for n in range(num_images):
        # Plot the image
        axes[n, 0].imshow(load_img(imglist[n], target_size=(299, 299)))
        axes[n, 0].axis('off')
        axes[n, 0].set_title(imglist[n].split(os.path.sep)[-2])

        # Plot the predictions
        bar_plot = axes[n, 1].bar(range(len(classes)), predictions[n])
        axes[n, 1].set_xticks(range(len(classes)))
        axes[n, 1].set_xticklabels(classes, rotation=15, ha="right")
        axes[n, 1].set_title(
            f"Model prediction: {classes[np.argmax(predictions[n])]}")

        # Add text labels for predicted probabilities
        for i, v in enumerate(predictions[n]):
            if v > 0.004:
                axes[n, 1].text(i, v, f"{v:.2f}", ha="center")

    plt.savefig(os.path.join(output, 'predictedBars.png'), bbox_inches='tight')
    # plt.show()


def predictBarColor(model, image_path, classes, img_size=244, num_images=None, output='results'):
    """
    Predicts the class probabilities for a set of images using the specified model.

    Args:
        model (tensorflow.keras.Model): The model to use for prediction.
        image_path (str): The path to the directory/list containing the images to predict.
        classes (list of str): A list of the class names in the order predicted by the model.
        img_size (int, optional): The target size to resize the images. Defaults to 244.
        num_images (int, optional): The number of images to plot the predictions for.

    Returns:
        None
    """

    CreateDir(output)
    imglist = get_image_paths(image_path)
    num_images = num_images or len(imglist)
    images = [img_to_array(load_img(img_path, target_size=(
        img_size, img_size))) / 255.0 for img_path in imglist[:num_images]]

    # Predict the classes for the images
    predictions = model.predict(np.array(images))

    # Define colors for each class
    num_classes = len(classes)
    colors = cm.rainbow(np.linspace(0, 1, num_classes))

    # Plot the images and predictions
    fig, axes = plt.subplots(num_images, 2, figsize=(16, 4 * num_images))
    fig.subplots_adjust(hspace=0.2, wspace=-0.2)
    # fig.subplots_adjust(hspace=0.6, wspace=-0.2)

    for n in range(num_images):
        # Plot the image
        axes[n, 0].imshow(load_img(imglist[n], target_size=(299, 299)))
        axes[n, 0].axis('off')
        axes[n, 0].set_title(imglist[n].split(os.path.sep)[-2])

        # Plot the predictions
        bar_plot = axes[n, 1].bar(
            range(len(classes)), predictions[n], color=colors)
        axes[n, 1].set_xticks(range(len(classes)))
        axes[n, 1].set_xticklabels(range(len(classes)))
        # axes[n, 1].set_xticklabels(classes, rotation=20, ha="right")
        axes[n, 1].set_title(
            f"Model prediction: {classes[np.argmax(predictions[n])]}")

        # Add text labels for predicted probabilities
        for i, v in enumerate(predictions[n]):
            if v > 0.004:
                axes[n, 1].text(i, v, f"{v:.2f}", ha="center")

        # Create legend with class names and corresponding colors
        legend_handles = [mpatches.Patch(
            color=colors[i], label=classes[i]) for i in range(len(classes))]
        axes[n, 1].legend(handles=legend_handles,
                          loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.savefig(os.path.join(output, 'predictedBarsColor.png'),
                bbox_inches='tight')
    # plt.show()




def plot_probability_distribution_and_roc_curves_one_vs_rest(model_multiclass, data_generator, steps, output='results'):
    '''
    Plots the probability distributions and the ROC curves for each class of a one vs rest multiclass classification model.

    Parameters:
    model_multiclass: A multiclass classification model.
    data_generator: A data generator that generates test data.
    steps: Number of steps to run the generator for.

    Returns:
    None
    '''

    CreateDir(output)
    plt.figure(figsize=(12, 8))
    bins = [i / 20 for i in range(20)] + [1]
    classes = model_multiclass.classes_
    roc_auc_ovr = {}

    # Generate X_test and y_test using the data generator
    X_test = []
    y_test = []
    for i in range(steps):
        batch = next(data_generator)
        X_test.extend(batch[0])
        y_test.extend(batch[1])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Get y_proba from the model
    y_proba = model_multiclass.predict_proba(X_test)

    for i in range(len(classes)):
        # Gets the class
        c = classes[i]

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame(X_test.copy(), columns=[
                              'feature_{}'.format(i) for i in range(X_test.shape[1])])
        df_aux['class'] = [1 if y == c else 0 for y in y_test]
        df_aux['prob'] = y_proba[:, i]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 3, i + 1)
        sns.histplot(x="prob", data=df_aux, hue='class',
                     color='b', ax=ax, bins=bins)
        ax.set_title(c)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 3, i + 4)
        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
        plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
        ax_bottom.set_title("ROC Curve OvR")

        # Calculates the ROC AUC OvR
        roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])

    plt.tight_layout()



