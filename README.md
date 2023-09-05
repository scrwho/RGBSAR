# RGBSAR

Welcome to the RGBSAR repository! This repository contains code and resources related to RGB and SAR image processing, fusion, and machine learning.

## Project Overview

The project is organized into the following sections:

### Installation

To set up the necessary modules for this project, please refer to the [requirements.txt](requirements.txt) file.

Before running the project, make sure to install the required Python modules. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

### Libraries

You can find a list of imported libraries used in the project in [Libraries.ipynb](Notebook/Libraries.ipynb).

### Functions

Project functions are documented in two notebooks: [Functions.ipynb](Notebook/Functions.ipynb) and [ModelsListDiffFuntions.ipynb](Notebook/ModelsListDiffFuntions.ipynb).

### Exploratory Data Analysis (EDA)

- [analysis.ipynb](Notebook/analysis.ipynb): Provides an overview of the project's workflow.
- [DeBlur.ipynb](Notebook/DeBlur.ipynb): Detects and corrects blurry images.
- [EDA.ipynb](Notebook/EDA.ipynb): Generates a dataframe with image properties.
- [GenerateMoreImages.ipynb](Notebook/GenerateMoreImages.ipynb): Creates synthetic data from a limited set of images.
- [GetSysInfo.ipynb](Notebook/GetSysInfo.ipynb): Retrieves system information such as OS details, CPU, and GPU.
- [resize_images.ipynb](Notebook/resize_images.ipynb): Resizes images to a uniform size.

### Image Data Fusion

- [FusionRGBSAR.ipynb](Notebook/FusionRGBSAR.ipynb): Performs image fusion for pairs of SAR and RGB images.

### Model Training, Evaluation, and Predictions

- [PlotHists.ipynb](Notebook/PlotHists.ipynb): Plots training history for models.
- [ModelsListDiffTraining.ipynb](Notebook/ModelsListDiffTraining.ipynb): Trains image datasets using custom-built CNN models with pre-trained CNN models.
- [ModelsListDiffTestEvaluate.ipynb](Notebook/ModelsListDiffTestEvaluate.ipynb): Evaluates the performance of trained models, performs predictions, and evaluates their effectiveness.
- [Modindivi.ipynb](Notebook/Modindivi.ipynb): Makes predictions using trained models on resource-limited operating systems.

### Equivalent Python Scripts

Python script equivalents for the notebooks can be found in the [Scripts](Scripts) folder.

<!-- ## Project Status -->

<!-- The project is under active development, and more content will be uploaded soon. -->

<!-- If you have any questions or suggestions, feel free to reach out! -->