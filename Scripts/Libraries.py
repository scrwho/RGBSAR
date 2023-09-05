##!/usr/bin/env python
# coding: utf-8

import netCDF4 as nc
from PIL import ImageFont
from netCDF4 import Dataset
from torchvision.models.vgg import vgg19
from scipy import linalg as la
import logging
import json
import uuid
import socket
import requests
import hashlib
import platform
import warnings
import sporco
import pywt
import splitfolders
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import glob
from multiprocessing import Pool, cpu_count
from functools import lru_cache
from pathlib import PureWindowsPath, WindowsPath
from skimage.transform import resize
from skimage.filters import laplace
from scipy.ndimage import variance
import seaborn as sns
from cv2 import IMREAD_COLOR, IMREAD_UNCHANGED
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from keras.utils.np_utils import to_categorical
from keras_sequential_ascii import keras2ascii
import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.applications import InceptionV3, ResNet50, ResNet101, InceptionResNetV2, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.utils import img_to_array, load_img, plot_model
from tensorflow.keras import optimizers, regularizers, mixed_precision, backend as K
from tensorflow.keras.layers import *
import tensorflow as tf
from pycore.tikzeng import *
from pycore.blocks import *
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import graphviz
import netCDF4
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
import sys
import time
import pickle
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFile
from IPython.display import display
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')


sns.set_style('whitegrid')
