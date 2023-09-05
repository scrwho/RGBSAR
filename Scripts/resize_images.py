##!/usr/bin/env python
# coding: utf-8


# import nbimporter
# import import_ipynb
from tqdm import tqdm
import ModelsListDiffFuntions
from ModelsListDiffFuntions import *
from Functions import *
import Functions
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
print(dir(Functions))


# print(dir(ModelsListDiffFuntions))


# C:\Users\scrwh\Documents\PythonScripts\Master_Thesis\Function\Synthetic_data\downloaded_images
data_folder = os.path.join('..', '..', 'Master_Thesis',
                           'Function', 'Synthetic_data', 'downloaded_images')
output_folder = os.path.join(
    '..', '..', 'Master_Thesis', 'Function', 'Synthetic_data', 'downloaded_img')


FolderTree(data_folder)


get_file_list(data_folder)


def resize_images(input_dir, new_size=244, output_dir='resized_images', quality=95, allowed_extensions=None):
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
            if img.mode == "P":
                img = img.convert("RGB")
            wpercent = (new_size / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            resized_img = img.resize(
                (new_size, new_size), resample=Image.Resampling.LANCZOS)

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
                          new_size=new_size, allowed_extensions=allowed_extensions)


data_folder = os.path.join('..', '..', 'Master_Thesis', 'data', 'data0')
output_folder = os.path.join('..', '..', 'Master_Thesis', 'data', 'data0b')


resize_images(data_folder, new_width=244)
