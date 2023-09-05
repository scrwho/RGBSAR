##!/usr/bin/env python
# coding: utf-8

# # Read Image Data


# Get all the images to be processed into a list
import sys
import splitfolders
import time
import matplotlib.pyplot as plt
from torchvision.models.vgg import vgg19
from scipy import linalg as la
import torch
import sporco
import re
import random
import pywt
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import os
from Functions import *
import Functions


def get_file_list(data_folder):
    # Walk through the directory structure and collect all files
    # Join the directory path and file name to create the complete file path
    return [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(data_folder) for file in files]

# Get file list for images only


def get_file_list_imgs(data_folder):
    # Walk through the directory structure and collect only image files with specific extensions
    return [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(data_folder) for file in files if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif')]


def get_image_paths(image_path):
    if isinstance(image_path, list):
        # If the input is already a list of image paths, use it as is
        image_paths = image_path
    elif os.path.isdir(image_path):
        # If the input is a directory, collect all image files within it
        image_paths = [os.path.join(root, file)
                       for root, dirs, files in os.walk(image_path)
                       for file in files
                       if file.endswith((".jpg", ".jpeg", ".png"))]
    else:
        # If the input is a single image file, use it as is
        image_paths = [image_path]

    return image_paths


# ## Check for Error


def copy_and_open(images, output_dir):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for img_path in images:
        img_path = os.path.abspath(img_path)
        img_name = os.path.basename(img_path)
        dest_path = os.path.join(output_dir, img_name)
        shutil.copy(img_path, dest_path)
    os.startfile(output_dir)


def display_random_images(images, n, saveDir='smapleImages', seed=1234):
    """
    Select n random images from a list of image file paths or a directory,
    and print their sizes as titles in a grid.
    """
    # If images is a directory, get all image file paths in the directory
    images = get_image_paths(images)

    # Select n random images from the list
    random.seed(seed)
    images = random.sample(images, n)
    copy_and_open(images, saveDir)
    n_cols = 3 if len(images) < 4 else 4
    n_rows = int(np.ceil(len(images) / n_cols))

    # Display the images in a grid with their sizes as titles
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5))
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    elif n_cols == 1:
        axs = np.expand_dims(axs, axis=1)
    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        axs[i // n_cols, i % n_cols].imshow(img)
        axs[i // n_cols, i % n_cols].set_title(f"{img.size[0]}x{img.size[1]}")
        axs[i // n_cols, i % n_cols].axis('off')
    for ax in axs.flat[len(images):]:
        ax.axis('off')
    saveDirResults = os.path.join(saveDir, 'results')
    os.makedirs(saveDirResults, exist_ok=True)
    figname = os.path.join(saveDir, 'results', 'sample_imge_size.png')
    plt.savefig(figname, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_blurred_images(images, threshold=100):
    """
    Given a directory or list of image file paths,
    returns a list of blurred image file paths.
    """
    # If images is a directory, get all image file paths in the directory
    images = get_image_paths(images)
    blurred_images = []
    blurred_count = 0
    # Loop through the images and check if they are blurred
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:  # Skip if image could not be read
            continue
        # Calculate the variance of Laplacian of the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        # If the variance is below a threshold, consider the image blurred
        if fm < threshold:
            blurred_images.append(img_path)
            blurred_count += 1
    return blurred_images, blurred_count


def remove_blur(input_dir, output_dir='deblurred', threshold=30, kernel_size=(5, 5), sharpening=True, allowed_extensions=None):
    """
    Remove blur from images in the input directory and save them in the output directory.

    Arguments:
    input_dir -- Input directory containing the images to be deblurred
    output_dir -- Output directory to save the deblurred images
    threshold -- Threshold for detecting blur (default: 30)
    kernel_size -- Kernel size for Gaussian blur (default: (5, 5))
    sharpening -- Apply sharpening after deblurring (default: True)
    allowed_extensions -- Set of allowed file extensions for deblurring (default: ['.jpg', '.jpeg', '.png'])
    """

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # input_dir = get_image_paths(input_dir) #os.path.abspath(input_dir)#.replace("\\", "/")
    # output_dir = os.path.abspath(output_dir).replace("\\", "/")

    if allowed_extensions is None:
        allowed_extensions = set(['.jpg', '.jpeg', '.png'])

    images = get_image_paths(input_dir)
    deblurred_images = []

    for entry in images:
        image = cv2.imread(entry)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        if fm < threshold:
            blurred = cv2.GaussianBlur(image, kernel_size, 0)
            deblurred = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

            if sharpening:
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                deblurred = cv2.filter2D(deblurred, -1, kernel)

            # Save deblurred image in the corresponding subdirectory of the output directory
            output_subdir = os.path.join(output_dir, os.path.dirname(entry))
            os.makedirs(output_subdir, exist_ok=True)
            output_file_path = os.path.join(
                output_subdir, os.path.basename(entry))

            cv2.imwrite(output_file_path, deblurred)
            deblurred_images.append(output_file_path)
        else:
            # If image is not blurred, copy it to the output directory with the same folder structure
            output_subdir = os.path.join(
                output_dir, os.path.dirname(entry)[len(input_dir):])
            os.makedirs(output_subdir, exist_ok=True)
            output_file_path = os.path.join(
                output_subdir, os.path.basename(entry))
            shutil.copy(entry.path, output_file_path)

    return deblurred_images


def resize_images(input_dir, output_dir, quality=95, new_width=244, allowed_extensions=None):
    """
    Resize images in the input directory and save them in the output directory.

    Arguments:
    input_dir -- Input directory containing the images to be resized
    output_dir -- Output directory to save the resized images
    quality -- JPEG quality for saving the resized images (default: 95)
    new_width -- Width in pixels for resizing the images (default: 244)
    allowed_extensions -- Set of allowed file extensions for resizing (default: ['.jpg', '.jpeg', '.png'])
    """
    input_dir = os.path.abspath(input_dir).replace("\\", "/")
    output_dir = os.path.abspath(output_dir).replace("\\", "/")

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if allowed_extensions is None:
        allowed_extensions = set(['.jpg', '.jpeg', '.png'])

    images = [entry for entry in os.scandir(input_dir) if entry.is_file(
    ) and entry.name.lower().endswith(tuple(allowed_extensions))]
    with tqdm(total=len(images), desc="Resizing images", unit="file", leave=False) as pbar:
        for entry in images:
            output_file_path = os.path.join(output_dir, f"{entry.name}")

            # Check if the file already exists in the output directory
            if os.path.exists(output_file_path):
                pbar.update(1)
                continue

            # Resize the image
            img = Image.open(entry.path)
            wpercent = (new_width / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            resized_img = img.resize(
                (new_width, hsize), resample=Image.Resampling.LANCZOS)

            # Save the resized image in the corresponding subdirectory of the output directory
            resized_img.save(output_file_path, quality=quality)
            pbar.update(1)

    # Recursively call the function on subdirectories
    for entry in os.scandir(input_dir):
        if entry.is_dir():
            sub_input_dir = os.path.join(
                input_dir, entry.name).replace("\\", "/")
            sub_output_dir = os.path.join(
                output_dir, entry.name).replace("\\", "/")
            resize_images(sub_input_dir, sub_output_dir, quality=quality,
                          new_width=new_width, allowed_extensions=allowed_extensions)


# # Image Fusion

def signaltonoise(a, axis, ddof):
    """
    Compute the signal-to-noise ratio of an array.

    Args:
        a (ndarray): Input array.
        axis (int or None): Axis along which to compute the mean and standard deviation.
        ddof (int): Delta degrees of freedom.

    Returns:
        ndarray: Signal-to-noise ratio.

    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def lowpass(s, lda, npad):
    """
    Perform low pass filtering using Tikhonov filter.

    Args:
        s (ndarray): Input array.
        lda (float): Regularization parameter.
        npad (int): Number of pixels to pad.

    Returns:
        ndarray: Filtered array.

    """
    # return tikhonov_filter(s, lda, npad)
    return sporco.signal.tikhonov_filter(s, lda, npad)


def get_activation(model, layer_numbers, input_image):
    """
    Get the activations of specified layers in a given model.

    Args:
        model: Pretrained model.
        layer_numbers (list): List of layer numbers to retrieve activations from.
        input_image (Tensor): Input image.

    Returns:
        list: List of activation arrays.

    """
    outs = []
    out = input_image
    for i in range(max(layer_numbers) + 1):
        with torch.no_grad():  # Reduces memory usage and speeds up calculations
            out = model.features[i](out)
        if i in layer_numbers:
            outs.append(np.rollaxis(out.detach().cpu().numpy()[0], 0, 3))
    return outs


def c3(s):
    """
    Convert a 2D or 3D image to a 3D array and rotate.

    Args:
        s (ndarray): Input image.

    Returns:
        ndarray: Rotated 3D array.

    """
    if s.ndim == 2:
        s3 = np.dstack([s, s, s])
    else:
        s3 = s
    return np.rollaxis(s3, 2, 0)[None, :, :, :]


def l1_features(out):
    """
    Compute L1 norm of the given array and return a matrix with zero edges.

    Args:
        out (ndarray): Input array.

    Returns:
        ndarray: L1 norm array with zero edges.

    """
    h, w, d = out.shape
    a_temp = np.zeros((h + 2, w + 2))  # All edges of the Matrix have been zero

    l1_norm = np.sum(np.abs(out), axis=2)
    a_temp[1:h + 1, 1:w + 1] = l1_norm
    return a_temp


def Fusion_PCA(rgb, sar):
    """
    Performs image fusion using PCA (Principal Component Analysis) algorithm.

    Args:
        rgb (numpy.ndarray): RGB image dataset (matrix) to be fused.
        sar (numpy.ndarray): SAR image dataset (matrix) to be fused.

    Returns:
        numpy.ndarray: Fused image dataset (matrix).
    """
    # Converting Image data to numpy Array to be able to do necessary calculation
    a = np.array(rgb)
    b = np.array(sar)
    # getting Image dimensions
    temp1 = a.shape
    temp2 = b.shape
    # Starting PCA algorithm
    # creating matrix with both Images
    vector1 = np.reshape(a, temp1[0] * temp1[1], order='F')
    vector2 = np.reshape(b, temp2[0] * temp2[1], order='F')
    # Convolution of created matrix
    c = np.cov(vector1, vector2)
    # getting Eigenvalue and Eigenvector of this matrix
    d, v = la.eig(c)
    sum1 = np.sum(v, axis=0)
    # Calculating PCA
    if d[0] >= d[1]:
        pca = np.divide(v[:, 0], sum1[0])
    else:
        pca = np.divide(v[:, 1], sum1[1])
    # Creating fused image
    result = (pca[0] * rgb) + (pca[1] * sar)
    return result


def Fusion_DWT_db2(rgb, sar):
    """
    Performs image fusion using Discrete Wavelet Transform (DWT) with Daubechies filter (db2).

    Args:
        rgb (numpy.ndarray): RGB image dataset (matrix) to be fused.
        sar (numpy.ndarray): SAR image dataset (matrix) to be fused.

    Returns:
        numpy.ndarray: Fused image dataset (matrix).
    """
    # decomposing each image using Discrete Wavelet Transform (DWT) with Daubechies filter (db2)
    coefficients_1 = pywt.wavedec2(rgb, 'db2', level=2)
    coefficients_2 = pywt.wavedec2(sar, 'db2', level=2)
    # creating variables to be used
    coefficients_h = list(coefficients_1)
    # fusing the decomposed image data
    coefficients_h[0] = (coefficients_1[0] + coefficients_2[0]) * 0.5
    # creating variables to be used
    temp1 = list(coefficients_1[1])
    temp2 = list(coefficients_2[1])
    temp3 = list(coefficients_h[1])
    # fusing the decomposed image data
    temp3[0] = (temp1[0] + temp2[0]) * 0.5
    temp3[1] = (temp1[1] + temp2[1]) * 0.5
    temp3[2] = (temp1[2] + temp2[2]) * 0.5
    coefficients_h[1] = tuple(temp3)
    # Creating fused image by reconstructing the fused decomposed image
    result = pywt.waverec2(coefficients_h, 'db2')
    return result


def fusion_strategy(feat_a, feat_b, source_a, source_b, img, unit):
    """
    Weighted-averaging method for fusion of SAR and RGB images.
    Args:
        feat_a: Feature map of image A.
        feat_b: Feature map of image B.
        source_a: Source image A.
        source_b: Source image B.
        img: Image to be fused.
        unit: Unit size for fusion.

    Returns:
        Fused image.
    """
    m, n = feat_a.shape
    m1, n1 = source_a.shape[:2]
    weight_ave_temp1 = np.zeros((m1, n1))
    weight_ave_temp2 = np.zeros((m1, n1))
    weight_ave_temp3 = np.zeros((m1, n1))

    for i in range(1, m):
        for j in range(1, n):
            a1 = feat_a[i - 1:i + 1, j - 1:j + 1].sum() / 9
            a2 = feat_b[i - 1:i + 1, j - 1:j + 1].sum() / 9
            a3 = img[i - 1:i + 1, j - 1:j + 1].sum() / 9

            weight_ave_temp1[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = a1 / (
                a1 + a2 + a3)
            weight_ave_temp2[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = a2 / (
                a1 + a2 + a3)
            weight_ave_temp3[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = a3 / (
                a1 + a2 + a3)

    if source_a.ndim == 3:
        weight_ave_temp1 = weight_ave_temp1[:, :, None]
    source_a_fuse = source_a * weight_ave_temp1
    if source_b.ndim == 3:
        weight_ave_temp2 = weight_ave_temp2[:, :, None]
    source_b_fuse = source_b * weight_ave_temp2
    if img.ndim == 3:
        weight_ave_temp3 = weight_ave_temp3[:, :, None]
    source_img_fuse = img * weight_ave_temp3

    if source_a.ndim == 3 or source_b.ndim == 3 or img.ndim == 3:
        gen = np.atleast_3d(
            source_a_fuse) + np.atleast_3d(source_b_fuse) + np.atleast_3d(source_img_fuse)
    else:
        gen = source_a_fuse + source_b_fuse + source_img_fuse

    return gen


def fusion_strategy2(feat_a, feat_b, source_a, source_b, img1, img2, unit):
    """
    Fusion strategy 2 for fusion of SAR and RGB images.
    Args:
        feat_a: Feature map of image A.
        feat_b: Feature map of image B.
        source_a: Source image A.
        source_b: Source image B.
        img1: Image 1 to be fused.
        img2: Image 2 to be fused.
        unit: Unit size for fusion.

    Returns:
        Fused image.
    """
    m, n = feat_a.shape
    m1, n1 = source_a.shape[:2]
    weight_ave_temp1 = np.zeros((m1, n1))
    weight_ave_temp2 = np.zeros((m1, n1))
    weight_ave_temp3 = np.zeros((m1, n1))
    weight_ave_temp4 = np.zeros((m1, n1))

    for i in range(1, m):
        for j in range(1, n):
            a1 = feat_a[i - 1:i + 1, j - 1:j + 1].sum() / 9
            a2 = feat_b[i - 1:i + 1, j - 1:j + 1].sum() / 9
            a3 = img1[i - 1:i + 1, j - 1:j + 1].sum() / 9
            a4 = img2[i - 1:i + 1, j - 1:j + 1].sum() / 9

            weight_ave_temp1[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = a1 / (
                a1 + a2 + a3 + a4)
            weight_ave_temp2[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = a2 / (
                a1 + a2 + a3 + a4)
            weight_ave_temp3[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = a3 / (
                a1 + a2 + a3 + a4)
            weight_ave_temp4[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = a4 / (
                a1 + a2 + a3 + a4)

    if source_a.ndim == 3:
        weight_ave_temp1 = weight_ave_temp1[:, :, None]
    source_a_fuse = source_a * weight_ave_temp1
    if source_b.ndim == 3:
        weight_ave_temp2 = weight_ave_temp2[:, :, None]
    source_b_fuse = source_b * weight_ave_temp2
    if img1.ndim == 3:
        weight_ave_temp3 = weight_ave_temp3[:, :, None]
    source_img_fuse = img1 * weight_ave_temp3
    if img2.ndim == 3:
        weight_ave_temp4 = weight_ave_temp4[:, :, None]
    source_img2_fuse = img2 * weight_ave_temp4

    if source_a.ndim == 3 or source_b.ndim == 3 or img1.ndim == 3 or img2.ndim == 3:
        gen = np.atleast_3d(source_a_fuse) + np.atleast_3d(source_b_fuse) + np.atleast_3d(
            source_img_fuse) + np.atleast_3d(source_img2_fuse)
    else:
        gen = source_a_fuse + source_b_fuse + source_img_fuse + source_img2_fuse

    return gen


def fuse(rgb, sar, model=None):
    """
    Fuse SAR and RGB images using a fusion strategy.
    Args:
        rgb: RGB image.
        sar: SAR image.
        model: Pre-trained VGG19 model for feature extraction.

    Returns:
        Fused image.
    """

    rgb = cv2.imread(rgb)
    sar = cv2.imread(sar)

    if len((sar).shape) != 2:
        sar = cv.cvtColor(sar, cv.COLOR_BGR2GRAY)
    if len((rgb).shape) != 2:
        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

    npad = 16
    lda = 5
    rgb_low, rgb_high = lowpass(rgb.astype(np.float32) / 255, lda, npad)
    sar_low, sar_high = lowpass(sar.astype(np.float32) / 255, lda, npad)

    img = Fusion_DWT_db2(rgb.astype(np.float32) / 255, sar_high)
    img1 = Fusion_PCA(sar.astype(np.float32) / 255, sar_high)
    img2 = Fusion_PCA(rgb_high, rgb.astype(np.float32) / 255)
    snr = np.max(signaltonoise(img, axis=0, ddof=0))

    if model is None:
        model = vgg19(True)
    model.cuda().eval()
    relus = [2, 7, 12, 21]
    unit_relus = [1, 2, 4, 8]

    rgb_in = torch.from_numpy(c3(rgb_high)).cuda()
    sar_in = torch.from_numpy(c3(sar_high)).cuda()

    relus_rgb = get_activation(model, relus, rgb_in)
    relus_sar = get_activation(model, relus, sar_in)

    rgb_feats = [l1_features(out) for out in relus_rgb]
    sar_feats = [l1_features(out) for out in relus_sar]

    saliencies = []
    saliency_max = None

    if 10 < snr < 19:
        for idx in range(len(relus)):
            saliency_current = fusion_strategy(rgb_feats[idx], sar_feats[idx],
                                               rgb_high, sar_high, img, unit_relus[idx])
            saliencies.append(saliency_current)

            if saliency_max is None:
                saliency_max = saliency_current
            else:
                saliency_max = np.maximum(saliency_max, saliency_current)

    else:
        for idx in range(len(relus)):
            saliency_current = fusion_strategy2(rgb_feats[idx], sar_feats[idx],
                                                rgb_high, sar_high, img1, img2, unit_relus[idx])
            saliencies.append(saliency_current)

            if saliency_max is None:
                saliency_max = saliency_current
            else:
                saliency_max = np.maximum(saliency_max, saliency_current)

    if rgb_low.ndim == 3 or sar_low.ndim == 3:
        low_fused = np.atleast_3d(rgb_low) + np.atleast_3d(sar_low)
    else:
        low_fused = rgb_low + sar_low
    low_fused = low_fused / 2
    high_fused = saliency_max

    return low_fused + high_fused


def show_images(rgb_image, sar_image, fused_image):
    """
    Display RGB, SAR, and fused images using matplotlib.pyplot.

    Arguments:
    rgb_image -- RGB image as a numpy array or a string representing the image file path
    sar_image -- SAR image as a numpy array or a string representing the image file path
    fused_image -- Fused image as a numpy array or a string representing the image file path
    """

    # Check if the input images are numpy arrays or file paths
    if isinstance(rgb_image, str):
        # Read RGB image using cv2 if it is a file path
        rgb_image = cv2.imread(rgb_image)
    if isinstance(sar_image, str):
        # Read SAR image using cv2 if it is a file path
        sar_image = cv2.imread(sar_image, cv2.IMREAD_GRAYSCALE)
    if isinstance(fused_image, str):
        # Read fused image using cv2 if it is a file path
        fused_image = cv2.imread(fused_image)

    # Create a figure and set up subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    # Plot SAR image
    axes[1].imshow(sar_image)
    axes[1].set_title('SAR Image')
    axes[1].axis('off')

    # Plot fused image
    axes[2].imshow(fused_image, cmap='gray')
    axes[2].set_title('Fused Image')
    axes[2].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()


# # Image data and Model implementation
# ## Split Image data into Train, Validate, Test


# Define the output folder name
output_folder = "data3"

root = sys.path[0]

# Create the full path for the output folder
data_folder = os.path.join(root, output_folder)

# Define the input folder path
# /d/Jacobs/Semester4/Master_Thesis/Acquahmeyer/Sample_Dataset/Sar and RGB/archive/fused_image
input_folder = os.path.join("D", os.sep, "Jacobs", "Semester4", "Master_Thesis",
                            "Acquahmeyer", "Sample_Dataset", "Sar and RGB", "archive", "fused_image")

# Create the output folder if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

# Print the current working directory
current_directory = os.getcwd()
print("Current Directory:", current_directory)

# Split the input_folder into training, validation, and testing sets with a ratio of 70%, 20%, and 10% respectively
splitfolders.ratio(
    input_folder,
    output=data_folder,
    seed=1337,
    ratio=(0.7, 0.2, 0.1),
    group_prefix=None,
)


# ## Build the models


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


def custom_model(data_dir, batch_size=32, epochs=50, img_size=244, num_classes=9):

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


def evaluate_models(data_dir, batch_size=32, epochs=50, img_size=244):
    # Generate the folder path with the current date and time
    foldername = os.path.join(
        'Models', 'TrainingData').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    folder_path = f'{foldername}_{now}'
    os.makedirs(folder_path, exist_ok=True)

    # Check if a GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("GPU available, training on GPU...")
        device_name = tf.test.gpu_device_name()
    else:
        print("GPU not available, training on CPU...")
        device_name = "/CPU:0"

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Data augmentation for the validation set
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training, validation, and test sets
    train_generator = load_data_generator(
        train_datagen, data_dir, 'train', img_size=img_size, batch_size=batch_size)
    val_generator = load_data_generator(
        val_datagen, data_dir, 'val', img_size=img_size, batch_size=batch_size)
    test_generator = load_data_generator(
        val_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    # Get the number of classes and labels
    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    # Model creation
    models = [('Inception-V2', InceptionV3, 'imagenet'), ('ResNet-50', ResNet50, 'imagenet'),
              ('ResNet-101', ResNet101, 'imagenet'), ('Inception-ResNet-V2',
                                                      InceptionResNetV2, 'imagenet'),
              ('VGG16', VGG16, 'imagenet'), ('Custom', custom_model, None)]

    model_results = {'Model': [], 'Training Accuracy': [],
                     'Validation Accuracy': [], 'Test Accuracy': []}

    for model_name, model_fn, weights in models:
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
                f'{folder_path}/best_{model_name}_model.h5', monitor='val_loss', save_best_only=True)

            # Train the model
            print(f'Training Model: {model_name}')
            start_time = datetime.now()
            print(f'{model_name} Started: {start_time}')
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=len(val_generator),
                callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
            )
            end_time = datetime.now()
            print(f'{model_name} Ended: {end_time}')
            print(f'Duration: {end_time - start_time}')

            # Run the garbage collector to free memory
            gc.collect()

            now = datetime.now().strftime('%Y-%m-%d %H%M')
            # Save the model and its history to disk
            model.save(
                f'{folder_path}/wind_turbine_{model_name}_model_{now}.h5')
            with open(f'{folder_path}/wind_turbine_{model_name}_history_{now}.pkl', 'wb') as f:
                pickle.dump(history.history, f)

        # Evaluation
        _, train_acc = model.evaluate(train_generator)
        _, val_acc = model.evaluate(val_generator)
        _, test_acc = model.evaluate(test_generator)
        model_results['Model'].append(model_name)
        model_results['Training Accuracy'].append(train_acc)
        model_results['Validation Accuracy'].append(val_acc)
        model_results['Test Accuracy'].append(test_acc)

    # Saving the performance in a DataFrame
    df = pd.DataFrame(model_results)
    df.set_index('Model', inplace=True)
    df['Average'] = df.select_dtypes(include='number').mean(axis=1)

    # Generate the filename with the current date and time
    file_name = os.path.join(
        folder_path, 'TrainingData').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    filename = f'{file_name}_{now}.csv'

    # Save the DataFrame to the CSV file
    df.to_csv(filename, index=True)

    return df


# ## Model Results and Evaluation
# ### Training History for all the models


def plot_training_histories(histories, model_names):
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
    plt.savefig('training_histories.png', bbox_inches='tight')
    plt.show()


# ## Evaluation results of the models


def EvaluateTest(data_dir, model, history, batch_size=32, img_size=256):

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
    df.to_csv("EvaluateTime_report.csv", index=True)

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
    df.to_csv('history.csv', index=True)
    # print the dataframe
    print(df, '\n')

    plot_training_history(history)

    plot_training_historyB(history)

    classNmaes = list(test_generator.class_indices.keys())

    # Evaluate the model on the test set
    print('\n Evaluate the model on the test set')
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
    plt.savefig('auc_roc_scores.png', bbox_inches='tight')
    plt.show()

    print("clases: \n", class_names)
    # Compute the classification report
    predicted_labels = np.argmax(predicted_labels, axis=1)
    cm = confusion_matrix(true_labels, predicted_labels)

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm = df_cm.round(2)
    df_cm.to_csv('confusion_matrix.csv', index=True)

    cr = classification_report(
        true_labels, predicted_labels, target_names=class_names)
    cr_dict = classification_report(
        true_labels, predicted_labels, target_names=class_names, output_dict=True)
    df_cr = pd.DataFrame(cr_dict).transpose()
    df_cr = df_cr.round(2)
    df_cr.to_csv("classification_report.csv", index=True)

    print(" \nConfusion matrix:\n", cm)
    print(df_cm)

    # plot_confusion_matrix(cm, class_names)
    plot_confusion_matrixN(cm, class_names, normalize=True)

    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('Ground Truth label')
    name = "Confusion_matrix_normalized.png"
    plt.savefig(name)
    # plt.show()

    print(" \nClassification report:\n", cr)
    print(df_cr)

    # Create a table of AUC-ROC scores for each class
    auc_roc_table = pd.DataFrame(
        {'Class': class_names, 'AUC-ROC': auc_roc_scores})
    auc_roc_table = auc_roc_table.round(2)
    auc_roc_table.to_csv("auc_roc_table.csv", index=False)
    print("\nauc_roc_table:\n", auc_roc_table)

    lb = LabelBinarizer()
    true_labels_binary = lb.fit_transform(true_labels)
    predicted_labels_binary = lb.transform(predicted_labels)

    avg_precision = average_precision_score(
        true_labels_binary, predicted_labels_binary, average='macro')
    print("Average Precision Score: ", avg_precision)


# ## Detect Fault in sample or given set of images


def detect_faultsdf(model, image_path, class_names, img_size=256, threshold=0.2):
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
