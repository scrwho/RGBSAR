# coding: utf-8


# Computer Vision
from __future__ import unicode_literals
import ModelsListDiffFuntions
from ModelsListDiffFuntions import *
import requests
import hashlib
import warnings
import sporco
import pywt
import splitfolders
import import_ipynb
import nbimporter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import sklearn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from datetime import datetime
import time
from tqdm import tqdm
from scipy import linalg as la
import torch
import re
import glob
from io import StringIO
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import struct
import random
import platform
from PIL import Image, ImageFile
import subprocess
import shutil
from pathlib import Path, PureWindowsPath, WindowsPath
import os
from skimage.transform import resize
from skimage.filters import laplace
from skimage.color import rgb2gray
from skimage import io
from scipy.ndimage import variance
import cv2
from cv2 import IMREAD_COLOR, IMREAD_UNCHANGED

# Data Analysis and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('whitegrid')

# Image Processing
plt.ion()

# File and Directory Handling

# Image Processing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Utility Functions

# Machine Learning


# Miscellaneous


warnings.filterwarnings('ignore')


def add_path_to_sys(path):
    module_path = os.path.abspath(path)
    if module_path not in sys.path:
        sys.path.append(module_path)


usePath = os.path.join(r'c:', os.sep, 'Users', 'scrwh',
                       'Documents', 'PythonScripts')
add_path_to_sys(usePath)


# print(dir(ModelsListDiffFuntions))


def FolderTree(FolderPath):
    for root, dirs, files in os.walk(FolderPath):
        level = root.count(os.sep) - FolderPath.count(os.sep)
        indent = ' ' * 4 * level
        num_files = len(files)
        print(
            f"{indent}{os.path.basename(root)}/ ({num_files} file{'s' if num_files != 1 else ''})")


def get_file_list(data_folder):
    return [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(data_folder) for file in files]


def get_file_list_imgs(data_folder):
    return [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(data_folder) for file in files if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif')]


def random_sample(items, sample_size, seed=1234):
    """
    Returns a random sample of size sample_size from items, using the seed specified.

    Parameters:
    items (list): the list of items to sample from
    sample_size (int): the number of items to sample
    seed (int): the seed to use for the random number generator

    Returns:
    list: a random sample of size sample_size from items
    """
    # Set the seed for the random number generator
    random.seed(seed)

    # Sample the items
    return random.sample(items, sample_size)


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


def copy_and_open(images, output_dir):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for img_path in images:
        img_path = os.path.abspath(img_path)
        img_name = os.path.basename(img_path)
        dest_path = os.path.join(output_dir, img_name)
        shutil.copy(img_path, dest_path)
    os.startfile(output_dir)


def sep():
    return os.path.sep


graphics = {
    "InputLayer": "  |  ",
    "Convolution1D": " \|/ ",
    "Convolution2D": " \|/ ",
    "Convolution3D": " \|/ ",
    "Conv1D": " \|/ ",
    "Conv2D": " \|/ ",
    "Conv3D": " \|/ ",
    "Conv2DTranspose": " /|\ ",
    "SeparableConv2D": r" /|\x",
    "UpSampling1D": "AAAAA",
    "UpSampling2D": "AAAAA",
    "UpSampling3D": "AAAAA",
    "Cropping1D": " ||| ",
    "Cropping2D": " ||| ",
    "Cropping2D": " ||| ",
    "Activation": " f| ",
    "Flatten": "|||||",
    "MaxPooling1D": "Y max",
    "MaxPooling2D": "Y max",
    "MaxPooling3D": "Y max",
    "AveragePooling1D": "Y avg",
    "AveragePooling2D": "Y avg",
    "AveragePooling3D": "Y avg",
    "GlobalMaxPooling1D": "Y^max",
    "GlobalMaxPooling2D": "Y^max",
    "GlobalAveragePooling1D": "Y^avg",
    "GlobalAveragePooling2D": "Y^avg",
    "Dropout": " | ||",
    "Dense": "XXXXX",
    "ZeroPadding1D": "\|||/",
    "ZeroPadding2D": "\|||/",
    "ZeroPadding3D": "\|||/",
    "BatchNormalization": " μ|σ ",
    "Reshape": "  |  ",
    "Permute": "  |  ",
    "Embedding": "emb |",
    "LSTM": "LLLLL",
    "GRU": "LLLLL"
}


def jsonize(model):
    res = []
    for layer in model.layers:
        x = {}

        x["name"] = layer.name
        x["kind"] = layer.__class__.__name__
        x["input_shape"] = layer.input_shape[1:]
        x["output_shape"] = layer.output_shape[1:]
        x["n_parameters"] = layer.count_params()
        try:
            x["activation"] = layer.activation.__name__
        except AttributeError:
            x["activation"] = ""

        res.append(x)
    return res


def compress_layers(jsonized_layers):
    res = [jsonized_layers[0]]
    for each in jsonized_layers[1:]:
        if each["kind"] == "Activation" and res[-1]["activation"] in ["", "linear"]:
            res[-1]["activation"] = each["activation"]
        else:
            res.append(each)
    return res


# data_template = "{activation:>15s}   #####   {shape} = {length}"
data_template = "{activation:>20s}   #####   {shape}"
layer_template = "{kind:>20s}   {graphics} -------------------{n_parameters:10d}   {percent_parameters:5.1f}%"


def product(iterable):
    res = 1
    for each in iterable:
        res *= each
    return res


def print_dim_tuple(t):
    if len(t) > 1:
        return " ".join(["{:4d}".format(x) for x in t])
    else:
        return "{:9d}".format(t[0])


def print_layers(jsonized_layers, sparser=False, simplify=False, header=True):

    if simplify:
        jsonized_layers = compress_layers(jsonized_layers)

    all_weights = sum([each["n_parameters"] for each in jsonized_layers])

    if header:
        print("           OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)\n")

    print(data_template.format(
        activation="Input",
        shape=print_dim_tuple(jsonized_layers[0]["input_shape"]),
        # length=product(jsonized_layers[0]["output_shape"])
    ))

    for each in jsonized_layers:

        if sparser:
            print("")

        print(layer_template.format(
            kind=each["kind"] if each["kind"] != "Activation" else "",
            graphics=graphics.get(each["kind"], "?????"),
            n_parameters=each["n_parameters"],
            percent_parameters=100 * each["n_parameters"] / all_weights
        ))

        if sparser:
            print("")

        print(data_template.format(
            activation=each["activation"] if each["activation"] != "linear" else "",
            shape=print_dim_tuple(each["output_shape"]),
            # length=product(each["output_shape"])
        ))


def sequential_model_to_ascii_printout(model, sparser=False, simplify=True, header=True):
    print_layers(jsonize(model), sparser=sparser,
                 simplify=simplify, header=header)


def loadImages(path):
    '''Put files into lists and return them as one list with all images 
     in the folder'''
    image_files = sorted([os.path.join(path, 'train', file)
                          for file in os.listdir(path + "/train")
                          if file.endswith((".jpg", ".jpeg", ".png"))])
    return image_files

# Display two images


def display(a, b, title1="Original", title2="Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()

# Display one image


def display_one(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.show()


# Preprocessing
def processing(data, height=220, width=220):

    # Reading 3 images to work
    img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:3]]
    try:
        print('Original size', img[0].shape)
    except AttributeError:
        print("shape not found")

    # --------------------------------
    # setting dim of the resize
    # height = 220
    # width = 220
    dim = (width, height)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

    # Checcking the size
    try:
        print('RESIZED', res_img[1].shape)
    except AttributeError:
        print("shape not found")

    # Visualizing one of the images in the array
    original = res_img[1]
    display_one(original)
    # ----------------------------------
    # Remove noise
    # Using Gaussian Blur
    no_noise = []
    for i in range(len(res_img)):
        blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)
        no_noise.append(blur)

    image = no_noise[1]
    display(original, image, 'Original', 'Blured')
    # ---------------------------------
    # Segmentation
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Displaying segmented images
    display(original, thresh, 'Original', 'Segmented')
    # Further noise removal (Morphology)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Displaying segmented back ground
    display(original, sure_bg, 'Original', 'Segmented Background')

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    # Displaying markers on the image
    display(original, markers, 'Original', 'Marked')
