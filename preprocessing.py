import numpy as np
from scipy import  ndimage
from skimage import io, exposure, transform
import pydicom

# Read the DICOM image
def load_dicom(file_path):
    return pydicom.dcmread(file_path).pixel_array

# Denoising using median filtering
def denoise_image(image):
    return  ndimage.median_filter(image, size=3)

# Resampling to a target size
def resample_image(image, target_shape=(512,  512)):
    return transform.resize(image, target_shape)

# Intensity normalization to [0,  1]
def normalize_image(image):
    max_val = np.max(image)
    min_val = np.min(image)
    return (image - min_val) / (max_val - min_val)

# Load a DICOM image
image_path = "path_to_your_dicom_image.dcm"
image = load_dicom(image_path)

# Denoise the image
denoised_image = denoise_image(image)

# Resample the image
resampled_image = resample_image(denoised_image)

# Normalize the image
normalized_image = normalize_image(resampled_image)

# Now normalized_image is ready for machine learning