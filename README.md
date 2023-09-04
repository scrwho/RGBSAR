# RGBSAR

The workflow, data and codes will be uploaded soon.

Basic codes has been uploaded.
Code is being clean to make it presentable# RGBSAR2

Install the modules from the [requirements](requirements.txt) text file

## Libraries
List of imported Libraries can be found in [Libraries.ipynb](Libraries.ipynb)

## Functions for the projects.
these are in [Functions.ipynb](Functions.ipynb) and [ModelsListDiffFuntions.ipynb](ModelsListDiffFuntions.ipynb)

## Workflow for EDA
[analysis.ipynb](analysis.ipynb): Summary of the workflow
[DeBlur.ipynb](DeBlur.ipynb): Check for blur Images and fix them
[EDA.ipynb](EDA.ipynb): Generate Dataframe of image properties
[GenerateMoreImages.ipynb](GenerateMoreImages.ipynb) : Generate Synthetic Data from the few images available
[GetSysInfo.ipynb](GetSysInfo.ipynb): Retrive the host system properties like OS name, OS version, CPU, GPU
[resize_images.ipynb](resize_images.ipynb): Resize images to a uniform Sizes

## Image Data Fusion
[FusionRGBSAR.ipynb](FusionRGBSAR.ipynb): Perform Image Fusion for a pair of SAR and RGB images

##  Model Training, Evaluation and predictions
[PlotHists.ipynb](PlotHists.ipynb): Plot model training history
[ModelsListDiffTraining.ipynb](ModelsListDiffTraining.ipynb): Performs training of Image dataset using custom built CNN model with Pretrain CNN Models
[ModelsListDiffTestEvaluate.ipynb](ModelsListDiffTestEvaluate.ipynb): Evaluate the performance of the trained models. Preform Predictions and evaluate
[Modindivi.ipynb](Modindivi.ipynb): Perform a prediction of the trained model on resource limited OS