##!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
from ModelsListDiffFuntions import *
import ModelsListDiffFuntions
import nbimporter
import import_ipynb


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


BlurImgs = os.path.join('blur_test_result', 'Sampleblurry')

BlurImges = get_file_list(BlurImgs)
BlurImges


plot_image_grid(BlurImges)


# ## Remove Blur


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


os.path.abspath('deblurred').replace("\\", "/")


debluImages = remove_blur(BlurImges)


plot_image_grid(debluImages)
