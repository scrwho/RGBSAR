##!/usr/bin/env python
# coding: utf-8


import seaborn as sns
from pandasql import sqldf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from Functions import *
import Functions
# import nbimporter
# import import_ipynb


import os
import sys


def add_path_to_sys(path):
    module_path = os.path.abspath(path)
    if module_path not in sys.path:
        sys.path.append(module_path)


usePath = os.path.join(r'c:', os.sep, 'Users', 'scrwh',
                       'Documents', 'PythonScripts')
add_path_to_sys(usePath)


sepp = os.path.sep

testDir = os.path.join('..', 'Synthetic_data',
                       'Fault_types - Copy').replace(os.path.sep, '/')
# C:\Users\scrwh\Documents\PythonScripts\Master_Thesis\Function\Synthetic_data\Fault_types - Copy
testDir


data_folder = os.path.join(
    '..', '..', 'data', 'data4d').replace(os.path.sep, '/')
data_folder


DataFileList = get_file_list_imgs(testDir)
len(DataFileList)


DataFileListCNN = get_file_list_imgs(data_folder)
len(DataFileListCNN)


def generate_image_df(imageList):
    df = pd.DataFrame()

    imageList = get_image_paths(imageList)

    paths = np.array(imageList)
    filenames = np.array([os.path.basename(path) for path in paths])
    directories = np.array([os.path.dirname(path) for path in paths])
    extensions = np.array([os.path.splitext(path)[1] for path in paths])
    split_classes = np.array(
        ['train' if 'train' in path else 'val' if 'val' in path else 'test' if 'test' in path else None for path in paths])
    type_classes = np.array([path.split(os.path.sep)[-2] for path in paths])

    df['path'] = paths
    df['filename'] = filenames
    df['directory'] = directories
    df['extension'] = extensions
    df['splitClass'] = split_classes
    df['typeClass'] = type_classes

    image_data = [cv2.imread(path) for path in paths]
    shapes = np.array([img.shape for img in image_data])

    df[['height', 'width', 'channels']] = shapes
    df['aspect_ratio'] = shapes[:, 1] / shapes[:, 0]

    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in image_data]
    blur_levels = np.array([cv2.Laplacian(img, cv2.CV_64F).var()
                           for img in gray_images])
    df['blur_level'] = blur_levels
    df['blurry'] = blur_levels < 30

    dpi = plt.gcf().dpi
    resolutions = shapes[:, 0] / dpi
    df['resolution'] = resolutions

    color_modes = np.array([img.shape[2] for img in image_data])
    df['color_mode'] = color_modes

    mean_intensities = np.array([img.mean() for img in image_data])
    df['mean_intensity'] = mean_intensities

    sharpness = np.array([cv2.Laplacian(img, cv2.CV_64F).var()
                         for img in gray_images])
    df['sharpness'] = sharpness

    threshold = 200
    proportions = np.array(
        [np.sum(img > threshold) / img.size for img in gray_images])
    df['above_threshold_prop'] = proportions

    color_props = []
    for img in image_data:
        if img.shape[2] == 3:
            red_prop = np.sum(img[:, :, 0]) / (img.size * 255)
            green_prop = np.sum(img[:, :, 1]) / (img.size * 255)
            blue_prop = np.sum(img[:, :, 2]) / (img.size * 255)
            color_props.append((red_prop, green_prop, blue_prop))
        else:
            # Append a list of None for grayscale images
            color_props.append([None, None, None])
    df[['red_prop', 'green_prop', 'blue_prop']] = color_props

    color_stats = np.array([cv2.meanStdDev(img) for img in image_data])
    df['mean_red'] = color_stats[:, 0, 2]
    df['mean_green'] = color_stats[:, 0, 1]
    df['mean_blue'] = color_stats[:, 0, 0]
    df['std_red'] = color_stats[:, 1, 2]
    df['std_green'] = color_stats[:, 1, 1]
    df['std_blue'] = color_stats[:, 1, 0]

    is_gray = np.array([len(img.shape) == 2 or img.shape[2] == 1 or (
        img[:, :, 0] == img[:, :, 1]).all() for img in image_data])
    df['grayscale'] = is_gray

    dominant_color = []
    for img in image_data:
        if img.shape[2] == 3:
            pixels = np.float32(img.reshape(-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            dominant_color.append(centers[0] / 255)
        else:
            dominant_color.append(None)
    df[['dominant_color_r', 'dominant_color_g', 'dominant_color_b']] = dominant_color

    brightness = np.array([cv2.mean(img)[0] for img in image_data])
    df['brightness'] = brightness

    contrast = []
    for img in gray_images:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        Q = hist_norm.cumsum()
        bins = np.arange(256)
        fn_min = np.inf
        for i in range(1, 256):
            p1, p2 = np.hsplit(hist_norm, [i])
            q1, q2 = Q[i], Q[255] - Q[i]
            if q1 == 0:
                q1 = 0.0000001
            if q2 == 0:
                q2 = 0.0000001
            b1, b2 = np.hsplit(bins, [i])
            m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
            v1, v2 = np.sum(((b1 - m1)**2) * p1) / \
                q1, np.sum(((b2 - m2)**2) * p2) / q2
            fn = v1 * q1 + v2 * q2
            if fn < fn_min:
                fn_min = fn
                thresh = i
        contrast.append(fn_min)
    df['contrast'] = contrast

    std_intensities = np.array([np.std(img) for img in image_data])
    df['std_intensity'] = std_intensities
    max_intensities = np.array([np.max(img) for img in image_data])
    df['max_intensity'] = max_intensities
    min_intensities = np.array([np.min(img) for img in image_data])
    df['min_intensity'] = min_intensities

    return df


start_times = datetime.now()
print(f'Started: {start_times}')

imagedf = generate_image_df(DataFileList)


end_times = datetime.now()
print(f'Ended: {end_times}')
print(f'Duration: {end_times - start_times}')

imagedf.head(5)


random_imagedf = imagedf.sample(n=10, random_state=42)
random_imagedf


column_names = imagedf.columns
column_names


start_times = datetime.now()
print(f'Started: {start_times}')

imagedfSplit = generate_image_df(DataFileListCNN)


end_times = datetime.now()
print(f'Ended: {end_times}')
print(f'Duration: {end_times - start_times}')


imagedfSplit


column_names = imagedfSplit.columns
column_names


imagedf.info


imagedf.describe().T


df = sqldf("SELECT * FROM imagedfSplit")
df


# Summary statistics table:
df.describe().T


sizes = imagedf.groupby(['height', 'width']).size(
).reset_index().rename(columns={0: 'count'})
sizes.plot.scatter(x='width', y='height')
plt.title('Image Sizes (pixels)')
# plt.title('Image Sizes (pixels) | {}'.format(Class)
plt.savefig('Size_distribution_of_image_data.png')


sizes = imagedfSplit.groupby(['height', 'width']).size(
).reset_index().rename(columns={0: 'count'})
sizes.plot.scatter(x='width', y='height')
plt.title('Image Sizes (pixels)')
# plt.title('Image Sizes (pixels) | {}'.format(Class)
plt.savefig('Size_distribution_of_image_data.png')


# Group the data by class
class_sizes = imagedf.groupby('typeClass')[['height', 'width']].mean()
# class_sizes = imagedfSplit.groupby('typeClass')[['height', 'width']].size().reset_index().rename(columns={0:'count'})

# Create a scatter plot for each class
for c in class_sizes.index:
    plt.scatter(class_sizes.loc[c, 'width'],
                class_sizes.loc[c, 'height'], label=c)

# Add labels and title to the plot
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.title('Image Sizes (pixels)')

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()


# Get unique classes
classes = imagedf['typeClass'].unique()

# Calculate number of rows needed for subplots
nrows = (len(classes) // 3) + (len(classes) % 3 > 0)

# Create figure and subplots
fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5*nrows))

# Iterate over each class and create a scatter plot
for i, Class in enumerate(classes):
    row, col = i // 3, i % 3
    class_df = imagedf[imagedf['typeClass'] == Class]
    sizes = class_df.groupby(['height', 'width']).size(
    ).reset_index().rename(columns={0: 'count'})
    axs[row, col].scatter(x=sizes['width'], y=sizes['height'])
    axs[row, col].set_title(
        'Image Sizes (pixels) | {} ({})'.format(Class, len(class_df)))
    axs[row, col].set_xlabel('Width')
    axs[row, col].set_ylabel('Height')

# Remove any unused subplots
for i in range(len(classes), nrows*3):
    row, col = i // 3, i % 3
    fig.delaxes(axs[row, col])

plt.tight_layout()
# plt.savefig('Size_distribution_of_image_data.png')
plt.savefig('Size_distribution_of_image_data_by_Class.png',
            bbox_inches='tight')
plt.show()


dff = sqldf("SELECT * FROM imagedfSplit")
dff


countSlit = sqldf(
    "SELECT splitClass, count(*) total FROM imagedfSplit group by splitClass")
countSlit.to_csv(os.path.join('results', 'SplitDistribution.csv'), index=False)
countSlit


a = """
SELECT splitClass, 
  COUNT(*) AS total,
  ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) total FROM imagedfSplit),1) AS "%"
FROM imagedfSplit 
GROUP BY splitClass
"""

countSlit = sqldf(a)

countSlit.to_csv(os.path.join('results', 'SplitDistribution.csv'), index=False)

countSlit


a = """ select case when r = 1 then split || ' : ' || CAST(split_Count AS TEXT) else '' end Split , class, total from
(select a.splitClass_a as split, a. total as split_Count, b.class, b.total, row_number() over(partition by splitClass_a order by splitClass_a,class) r  
            from 
            (SELECT splitClass as splitClass_a, count(*) total FROM imagedfSplit group by splitClass) a
            left join 
            (SELECT splitClass,typeClass as class, count(*) total FROM imagedfSplit group by splitClass,typeClass) b
            on a.splitClass_a = b.splitClass ) a
            
            """

countSlit = sqldf(a)
countSlit


a = """ select *, (train+val+test) total, ROUND((train+val+test) * 100.0 / (SELECT COUNT(*) FROM imagedfSplit), 1) AS "%"
from
(select a.*,b.val,c.test 
from
(SELECT typeClass as class, count(*) train FROM imagedfSplit where splitClass = 'train' group by typeClass ) a
left join 
(SELECT typeClass, count(*) val FROM imagedfSplit where splitClass = 'val' group by typeClass ) b
on a.class = b.typeClass
left join 
(SELECT typeClass, count(*) test FROM imagedfSplit where splitClass = 'test' group by typeClass ) c
on a.class = c.typeClass) a
group by class
            
            """

countClass = sqldf(a)
countClass.to_csv(os.path.join(
    'results', 'classDistribution.csv'), index=False)
countClass


# Group the data by the "train_val_test" column and count the number of occurrences
grouped_data = imagedfSplit.groupby("splitClass")["path"].count().reset_index()
grouped_data.columns = ["splitClass", "count"]

# Calculate the percentage of each "train_val_test" group
grouped_data["percentage"] = grouped_data["count"] / \
    grouped_data["count"].sum() * 100

# Create the barplot using seaborn
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=grouped_data, x="splitClass", y="count", ax=ax)

# Add the count values to the top of each bar
for i in ax.containers:
    ax.bar_label(i, label_type="edge", padding=10)

# Add the percentage values to the top of each bar (on the right side)
ax2 = ax.twinx()
sns.lineplot(data=grouped_data, x="splitClass",
             y="percentage", ax=ax2, color="red")
for index, row in grouped_data.iterrows():
    ax2.text(row.name, row.percentage, str(
        round(row.percentage)) + "%", ha="center", va="bottom")

# Set the axis labels and title
ax.set_xlabel("Train/Validation/Test")
ax.set_ylabel("Count")
ax.set_title("Distribution of Image Split")
plt.show()


# count of train_val_test
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

train_val_test_count = imagedfSplit.groupby('splitClass')['path'].count()

# ax1.bar(train_val_test_count.index, train_val_test_count.values, color='blue')
ax1.bar(train_val_test_count.index, train_val_test_count.values)
ax1.set_xlabel('splitClass')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Image Split (count)')

for i, v in enumerate(train_val_test_count.values):
    ax1.text(i, v, str(v), ha='center')

# percentage distribution of train_val_test
train_val_test_percentage = train_val_test_count / train_val_test_count.sum() * \
    100

# ax2.bar(train_val_test_percentage.index, train_val_test_percentage.values, color='grey')
ax2.bar(train_val_test_percentage.index, train_val_test_percentage.values)
ax2.set_xlabel('splitClass')
ax2.set_ylabel('Percentage')
ax2.set_title('Distribution of Image Split (percentage)')

for i, v in enumerate(train_val_test_percentage.values):
    ax2.text(i, v, f'{v:.1f}%', ha='center')

plt.show()


# Group the data by the "train_val_test" column and count the number of occurrences
grouped_data = imagedfSplit.groupby("typeClass")["path"].count().reset_index()
grouped_data.columns = ["typeClass", "count"]

# Calculate the percentage of each "train_val_test" group
grouped_data["percentage"] = grouped_data["count"] / \
    grouped_data["count"].sum() * 100

# Create the barplot using seaborn
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(data=grouped_data, x="typeClass", y="count", ax=ax)

# Rotate x-axis labels by 45 degrees
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   horizontalalignment='right')

# Add the count values to the top of each bar
for i in ax.containers:
    ax.bar_label(i, label_type="edge", padding=10)

# Add the percentage values to the top of each bar (on the right side)
ax2 = ax.twinx()

sns.lineplot(data=grouped_data, x="typeClass",
             y="percentage", ax=ax2, color="red")

for index, row in grouped_data.iterrows():
    ax2.text(row.name, row.percentage, str(
        round(row.percentage)) + "%", ha="center", va="bottom")

# Set the axis labels and title
ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.set_title("Images per Class")

plt.show()


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))


train_val_test_count = imagedfSplit.groupby('typeClass')['path'].count()

ax1.bar(train_val_test_count.index, train_val_test_count.values)
ax1.set_xlabel('class')
ax1.set_ylabel('Count')
ax1.set_title('Images per Class (count)')
ax1.set_xticklabels(train_val_test_count.index, rotation=45)

for i, v in enumerate(train_val_test_count.values):
    ax1.text(i, v, str(v), ha='center')

# percentage distribution of train_val_test
train_val_test_percentage = train_val_test_count / train_val_test_count.sum() * \
    100

ax2.bar(train_val_test_percentage.index, train_val_test_percentage.values)
ax2.set_xlabel('class')
ax2.set_ylabel('Percentage')
ax2.set_title('Images per Class (percentage)')
ax2.set_xticklabels(train_val_test_percentage.index, rotation=45)

for i, v in enumerate(train_val_test_percentage.values):
    ax2.text(i, v, f'{v:.1f}%', ha='center')

plt.show()


# Scatterplot of aspect ratio vs resolution:

sns.scatterplot(data=df, x='aspect_ratio', y='resolution')
plt.title('Aspect Ratio vs Resolution')


# plt.scatter(df['sharpness'], df['blurry'])
# plt.xlabel('Sharpness')
# plt.ylabel('Blurry (True/False)')
sns.scatterplot(data=df, x='sharpness', y='blurry')
plt.title('Sharpness vs Blurry')


# Table of average blurriness per class:

blurry_pct = df.groupby('typeClass')['blurry'].mean() * 100
blurry_pct = blurry_pct.rename('Percent Blurry').reset_index()
blurry_pct


# Violin plots of color channels:

sns.violinplot(data=df[['red_prop', 'green_prop', 'blue_prop']])
plt.title('Distribution of Color Channels')


# Heatmap of correlations:

sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Heatmap')


corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


numeric_cols = ['height', 'width', 'blur_level', 'resolution',
                'mean_intensity', 'sharpness', 'contrast', 'brightness']


sns.pairplot(df[numeric_cols], diag_kind='kde')
plt.show()


#
# ```python
# for col1 in numeric_cols:
#     for col2 in numeric_cols:
#         if col1 != col2:
#             plt.figure(figsize=(8, 4))
#             sns.scatterplot(x=col1, y=col2, data=df)
#             plt.title(f'Scatter Plot of {col1} vs. {col2}')
#             plt.show()
# ```

# ```python
# numeric_cols = ['height', 'width', 'aspect_ratio', 'blur_level', 'resolution', 'mean_intensity', 'sharpness', 'above_threshold_prop', 'red_prop', 'green_prop', 'blue_prop', 'mean_red', 'mean_green', 'mean_blue', 'std_red', 'std_green', 'std_blue', 'brightness', 'contrast', 'std_intensity', 'max_intensity', 'min_intensity']
#
# for col in numeric_cols:
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x=col, data=df)
#     plt.title(f'Box Plot of {col}')
#     plt.show()
# ```

# ```python
#
# # Function to generate summary statistics table
# def summary_statistics(df):
#     return df.describe().T
#
# # Function to generate histogram plots
# def histogram_plots(df, columns):
#     for column in columns:
#         plt.figure(figsize=(8, 6))
#         df[column].plot(kind='hist', bins=20, title=f'Histogram of {column}')
#         plt.xlabel(column)
#         plt.ylabel('Frequency')
#         plt.show()
#
# # Function to generate count of categorical values
# def count_categorical_values(df, columns):
#     for column in columns:
#         value_counts = df[column].value_counts()
#         print(f'Counts for {column}:\n{value_counts}\n')
#
# # Function to generate box plots
# def box_plots(df, numeric_columns):
#     for column in numeric_columns:
#         plt.figure(figsize=(8, 6))
#         df.boxplot(column=column)
#         plt.title(f'Box Plot of {column}')
#         plt.ylabel(column)
#         plt.show()
#
# # Function to generate correlation matrix
# def correlation_matrix(df):
#     correlation_matrix = df.corr()
#     return correlation_matrix
#
# # Function to generate scatter plots
# def scatter_plots(df, numeric_columns):
#     sns.pairplot(df[numeric_columns], diag_kind='kde')
#     plt.show()
#
# # Create a list of EDAs and their corresponding functions
# edas = [
#     {"name": "Summary Statistics", "function": summary_statistics},
#     {"name": "Histogram Plots", "function": histogram_plots, "args": [list(df.select_dtypes(include=['number']).columns)]},
#     {"name": "Count of Categorical Values", "function": count_categorical_values, "args": [list(df.select_dtypes(include=['object']).columns)]},
#     {"name": "Box Plots", "function": box_plots, "args": [list(df.select_dtypes(include=['number']).columns)]},
#     {"name": "Correlation Matrix", "function": correlation_matrix},
#     {"name": "Scatter Plots", "function": scatter_plots, "args": [list(df.select_dtypes(include=['number']).columns)]}
# ]
#
#
# # Execute the EDAs and generate tables and visualizations
# for eda in edas:
#     print(f"--- {eda['name']} ---")
#     if "args" in eda:
#         eda["function"](df, *eda["args"])
#     else:
#         result = eda["function"](df)
#         if isinstance(result, pd.DataFrame):
#             print(result)
# ```
