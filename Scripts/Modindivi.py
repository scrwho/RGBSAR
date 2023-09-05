##!/usr/bin/env python
# coding: utf-8


import os
import sys
import pickle
import random
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import platform


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelBinarizer
# from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import *

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
# from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, regularizers, mixed_precision
from tensorflow.keras.applications import InceptionV3, ResNet50, ResNet101, InceptionResNetV2, VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, UpSampling2D, concatenate, Activation,GlobalAveragePooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import *


import pandas as pd


def detect_faultsdfalone(model, image_path, class_names, img_size=256, threshold=0.2):
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
    num_images = len(image_paths)

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
    print(f"Percentage of wrong predictions: {wrong_predictions:.2f}%")

    # Save the DataFrame to the CSV file
    df.to_csv(
        f'SamplePredicted{platform.system()}{num_images}Immages.csv', index=False)

    # df.to_csv(f'SamplePredict{num_images}Immages.csv', index=False)

    # Create confusion matrix
    cm = confusion_matrix(
        df_split['Ground_Truth_Label'], df_split['Predicted_Label'])
    # Convert confusion matrix to a dataframe
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
    cm_df.to_csv(
        f'SamplePredictionsConfusionMatrix{platform.system()}{num_images}Immages.csv')

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cm, cmap='Blues')  # Change the color map
    ax.set_title('Confusion Matrix', fontsize=16)  # Increase title font size
    ax.set_xlabel('Predicted labels', fontsize=14)  # Increase label font size
    # Increase label font size
    ax.set_ylabel('Ground Truth labels', fontsize=14)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    # ax.set_xticklabels(class_names, fontsize=12)  # Decrease tick label font size
    # Decrease tick label font size and rotate xticklabels
    ax.set_xticklabels(class_names, fontsize=12, rotation=15, ha='right')
    # Decrease tick label font size
    ax.set_yticklabels(class_names, fontsize=12)

    # Add text to each cell with different font colors based on the background intensity
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            # Choose font color based on background intensity
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color=text_color, fontsize=14)

    plt.savefig(
        f'SamplePredictionsConfusionMatrix{platform.system()}{num_images}Immages.png', bbox_inches='tight')
    # plt.show()

    return df


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


def get_file_list(data_folder):
    return [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(data_folder) for file in files]


def get_image_paths(image_path):
    if isinstance(image_path, list):
        image_pathss = image_path
    elif os.path.isdir(image_path):
        image_pathss = [os.path.join(root, file)
                        for root, dirs, files in os.walk(image_path)
                        for file in files
                        if file.endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_pathss = [image_path]
    return image_pathss


# load the saved model
Inception_ResNet_V2_model = load_model(
    'Models/TrainingData_2023-03-06 0523/wind_turbine_Inception-ResNet-V2_model_2023-03-06 1602.h5')
# load the saved history
with open('Models/TrainingData_2023-03-06 0523/wind_turbine_Inception-ResNet-V2_history_2023-03-06 1602.pkl', 'rb') as f:
    Inception_ResNet_V2_history = pickle.load(f)


# In[4]:


data_folder = check_folder = os.path.join('..', 'Data', 'data4d')
check_folder = os.path.join('..', 'Data', 'data4c', 'test')
check_folderB = os.path.join('Synthetic_data', 'Fault_types - Copy (2)')

# C:\Users\scrwh\Documents\PythonScripts\Master_Thesis\Function\Synthetic_data\Fault_types - Copy (2)

print(len(check_folderB))
# C:\Users\scrwh\Documents\PythonScripts\Master_Thesis\data\data4b\check

batch_size = 32
epochs = 50
img_size = 244

# Retrieve input image size
# img_size = Inception_ResNet_V2_model.layers[0].input_shape[1]

val_datagen = ImageDataGenerator(
    rescale=1./255
)
# test_generator = load_data_generator(val_datagen, data_folder, 'test')
test_generator = load_data_generator(
    val_datagen, data_folder, 'test', img_size=img_size, batch_size=batch_size)


num_classes = len(test_generator.class_indices)
# labels = train_generator.class_indices
labels = list(test_generator.class_indices.keys())


# In[5]:


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


# In[6]:


all_filesc = get_file_list(check_folderB)
len(all_filesc)


# In[7]:


seed = 12345678901234567
# random.sample(all_filesc, 20)
imglist = get_random_samples(check_folderB, 100, seed)
# imglist


# In[8]:


df3 = detect_faultsdfalone(Inception_ResNet_V2_model,
                           imglist, labels, img_size)


# In[9]:


# df3


# imglist
# create an empty DataFrame with the desired columns
columns = ['Image', 'Ground Truth Label', 'Predicted Label',
           'Confidence', 'Prediction Time (sec)']
df = pd.DataFrame(columns=columns)

for i in range(1, 11):
    seed = None
    # random.sample(all_filesc, 20)
    imglist = get_random_samples(check_folderB, 100, seed)
    num_images = len(imglist)

    # call detect_faultsdfalone() and store the result in a temporary DataFrame
    temp_df = detect_faultsdfalone(
        Inception_ResNet_V2_model, imglist, labels, img_size)

    # add a new column 'runNum' with the value of i
    temp_df['runNum'] = i

    # append the temporary DataFrame to the main DataFrame
    df = pd.concat([df, temp_df], ignore_index=True)
    # df = df.append(temp_df, ignore_index=True)

df.to_csv(
    f'SamplePredictedSim{platform.system()}{num_images}Immages.csv', index=False)
print()
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
print(f"Percentage of wrong predictions: {wrong_predictions:.2f}%")


class_names = labels
# Create confusion matrix
cm = confusion_matrix(
    df_split['Ground_Truth_Label'], df_split['Predicted_Label'])
# Convert confusion matrix to a dataframe
cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
cm_df.to_csv(
    f'SamplePredictionsConfusionMatrixSim{platform.system()}{num_images}Immages.csv')

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cm, cmap='Blues')  # Change the color map
# Increase title font size
ax.set_title(
    f'Confusion Matrix of {total_Images} Images with {total_Predictions} predictions', fontsize=16)
ax.set_xlabel('Predicted labels', fontsize=14)  # Increase label font size
ax.set_ylabel('Ground Truth labels', fontsize=14)  # Increase label font size
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
# ax.set_xticklabels(class_names, fontsize=12)  # Decrease tick label font size
# Decrease tick label font size and rotate xticklabels
ax.set_xticklabels(class_names, fontsize=12, rotation=15, ha='right')
ax.set_yticklabels(class_names, fontsize=12)  # Decrease tick label font size

# Add text to each cell with different font colors based on the background intensity
for i in range(len(class_names)):
    for j in range(len(class_names)):
        # Choose font color based on background intensity
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color=text_color, fontsize=14)

plt.savefig(
    f'SamplePredictionsConfusionMatrixSim{platform.system()}{num_images}Immages.png', bbox_inches='tight')
