# RGBSAR

Welcome to the RGBSAR project repository. This repository contains code and data related to image processing, data analysis, and machine learning tasks for the RGBSAR project. Below is a brief overview of the project structure and key components.

## Project Overview

The RGBSAR project focuses on various aspects of image data processing and analysis. The repository includes the following components:

### Basic Codes

Some initial code snippets have been uploaded to the repository. These codes are currently being cleaned and organized to ensure better presentation and usability.

### Installation

Before running the project, make sure to install the required Python modules. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

The workflow, data and codes will be uploaded soon. 

Basic codes has been uploaded.

Code is being clean to make it presentable

## Getting Started

### Prerequisites

Install the modules from the [requirements](requirements.txt) text file

### Libraries

- List of imported Libraries can be found in [Libraries.ipynb](Notebook/Libraries.ipynb)

### Functions 

- These are in [Functions.ipynb](Notebook/Functions.ipynb) and [ModelsListDiffFuntions.ipynb](Notebook/ModelsListDiffFuntions.ipynb)

## Exploratory Data Analysis

### Workflow

- [analysis.ipynb](Notebook/analysis.ipynb): Summary of the workflow

### Preprocessing 

- [DeBlur.ipynb](Notebook/DeBlur.ipynb): Check for blur Images and fix them
- [EDA.ipynb](Notebook/EDA.ipynb): Generate Dataframe of image properties
- [GenerateMoreImages.ipynb](Notebook/GenerateMoreImages.ipynb) : Generate Synthetic Data from the few images available
- [GetSysInfo.ipynb](Notebook/GetSysInfo.ipynb): Retrive the host system properties like OS name, OS version, CPU, GPU
- [resize_images.ipynb](Notebook/resize_images.ipynb): Resize images to a uniform Sizes

## Image Data Fusion

- [FusionRGBSAR.ipynb](Notebook/FusionRGBSAR.ipynb): Perform Image Fusion for a pair of SAR and RGB images

## Model Training and Evaluation

- [PlotHists.ipynb](Notebook/PlotHists.ipynb): Plot model training history
- [ModelsListDiffTraining.ipynb](Notebook/ModelsListDiffTraining.ipynb): Performs training of Image dataset using custom built CNN model with Pretrain CNN Models
- [ModelsListDiffTestEvaluate.ipynb](Notebook/ModelsListDiffTestEvaluate.ipynb): Evaluate the performance of the trained models. Preform Predictions and evaluate
- [Modindivi.ipynb](Notebook/Modindivi.ipynb): Perform a prediction of the trained model on resource limited OS

The Eqivalent Python Script can be found in [Scripts](Scripts) Folder