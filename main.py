import os
import cv2
import numpy as np
import scipy.stats
from scipy import stats
import pandas as pd
from scipy.stats import skew, kurtosis


# Load images from the "data" Folder
data_folder = "data"
images = []
for filename in os.listdir(data_folder):
    img = cv2.imread(os.path.join(data_folder, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is not None:
        images.append(img)


def trim_mean(arr, percent):
    n = len(arr)
    k = int(round(n*(float(percent)/100)/2))
    return np.mean(arr[k+1:n-k])
def calculate_shannons_entropy(array):
    # Read the image in grayscale

    # Calculate the histogram of pixel intensities
    hist = cv2.calcHist([array], [0], None, [256], [0, 256])

    # Normalize the histogram to get the probability distribution
    hist_norm = hist / np.sum(hist)

    # Calculate Shannon's entropy
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

    return entropy


def calculate_image_kurtosis(array):

    # Flatten the image pixels to calculate kurtosis
    flat_image = array.flatten()

    # Calculate the kurtosis
    kurtosis_value = kurtosis(flat_image)

    return kurtosis_value

def calculate_image_skewness(array):

    # Flatten the image pixels to calculate kurtosis
    flat_image = array.flatten()

    # Calculate the kurtosis
    skewness_value = skew(flat_image)

    return skewness_value


# Initialize lists to store features
means = []
medians = []
max_vals = []
min_vals = []
trimmed_means = []
RMS= []
Variance = []
Standard_deviation = []
Percentile_1 = []
Percentile_50 = []
Percentile_75 = []
Percentile_99 = []
Interquartile = []
mean_Deviation = []
Kurtosis = []
Coefficient_of_variance = []
Shanon_entropy = []
Skewness = []
# Calculate features for each image
for img in images:
    means.append(np.mean(img))
    medians.append(np.median(img))
    max_vals.append(np.max(img))
    min_vals.append(np.min(img))
    trimmed_means.append(trim_mean(img,25))
    RMS.append(np.sqrt(np.mean(img**2)))
    Variance.append(np.var(img))
    Standard_deviation.append(np.std(img))
    Percentile_1.append(np.percentile(img,1))
    Percentile_50.append(np.percentile(img,50))
    Percentile_75.append(np.percentile(img,75))
    Percentile_99.append(np.percentile(img,99))
    Interquartile.append(np.percentile(img,75)-np.percentile(img,25))
    mean_Deviation.append(np.mean(np.absolute(img - np.mean(img))))
    Kurtosis.append(calculate_image_kurtosis(img))
    Coefficient_of_variance.append(np.std(img)/np.mean(img))
    Shanon_entropy.append(calculate_shannons_entropy(img))
    Skewness.append(calculate_image_skewness(img))
# Calculate correlation between features
features = np.array([means, medians, max_vals, min_vals, trimmed_means, RMS, Variance, Standard_deviation, Percentile_1
, Percentile_50, Percentile_75, Percentile_99, Interquartile, mean_Deviation, Kurtosis, Coefficient_of_variance
, Shanon_entropy, Skewness])
correlation_matrix = np.corrcoef(features)

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)