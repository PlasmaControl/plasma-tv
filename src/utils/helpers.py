import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.animation as animation

flatten = lambda x : x.reshape(len(x), -1)

def calculate_rms(array):
    """Calculate the root mean square of the array."""
    return np.sqrt(np.mean(np.square(array)))

def nearest_index(values, target):
    """Find the index of the nearest value in `values` to the given `target`."""
    return (np.abs(values - target)).argmin()

def convert_to_indices(radii, elevation, coordinates):
    """Convert physical coordinates to indices in the radii and elevation grids."""
    rad_idx = nearest_index(radii, coordinates[0])
    elev_idx = nearest_index(elevation, coordinates[1])
    return rad_idx, elev_idx

def crop_time(times, data, start_time, end_time):
    """Crop the data to the specified time range."""
    if data.ndim == 1:
        data = data.reshape(1, -1)
    start_idx = nearest_index(times, start_time)
    end_idx = nearest_index(times, end_time)
    return times[start_idx:end_idx], data[:,start_idx:end_idx]

def get_index(arr, coord):
    # Get the insertion indices
    ind = np.searchsorted(coord, arr)

    # Correct the indices to point to the nearest actual index
    ind = np.clip(ind, 0, len(coord) - 1)

    # Now, adjust the indices to get the closest value
    for i, cval in enumerate(arr):
        if ind[i] > 0 and abs(cval - coord[ind[i] - 1]) < abs(cval - coord[ind[i]]):
            ind[i] -= 1
            
    return ind

def save_data_to_pickle(data, filename, directory='outputs/inversion_data'):
    """
    Save the provided data to a pickle file in the specified directory.

    Parameters:
        data (dict): Dictionary containing the data to save.
        filename (str): The base filename for the pickle file.
        directory (str): The directory where the pickle file will be saved.
    """
    # Ensure the directory exists
    datpath = Path(directory)
    datpath.mkdir(parents=True, exist_ok=True)
    
    # Create the full path for the pickle file
    savepkl = datpath / f"{filename}.pkl"
    
    # Write the data to the pickle file
    with open(savepkl, 'wb') as file:
        pickle.dump(data, file)

    print(f"Data saved successfully to {savepkl}")
    
def reduce_res_2darr(arr, num):
    """Reduce a 2D array by averaging over blocks of size num."""
    return arr[:, ::num, ::num]

def match_image_to_histogram(image, target_cdf, bin_edges):
    src_hist, _ = np.histogram(image.ravel(), bins=bin_edges, density=False)
    src_cdf = np.cumsum(src_hist)
    src_cdf_normalized = src_cdf / src_cdf[-1]
    interp_values = np.interp(src_cdf_normalized, target_cdf, bin_edges[:-1])
    matched_image = np.interp(image.ravel(), bin_edges[:-1], interp_values)
    return matched_image.reshape(image.shape)

def match_images_to_histogram(image_array, target_cdf, bin_edges):
    M = image_array.shape[0]
    matched_images = np.empty_like(image_array)
    for i in range(M):
        matched_images[i] = match_image_to_histogram(image_array[i], target_cdf, bin_edges)
    return matched_images * 256

def normalize_image(image_array):
    means = np.mean(image_array, axis=(1, 2))
    stds = np.std(image_array, axis=(1, 2))
    return (image_array - means[:, None, None]) / stds[:, None, None]

def noisify(image):
    num,row,col = image.shape
    gauss = np.random.randn(num,row,col)
    gauss = gauss.reshape(num,row,col)        
    noisy = image + image * gauss * .1
    return noisy