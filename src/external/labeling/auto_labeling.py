import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.animation as animation

def calculate_rms(array):
    """Calculate the root mean square of the array."""
    return np.sqrt(np.mean(np.square(array)))

def find_nearest_index(values, target):
    """Find the index of the nearest value in `values` to the given `target`."""
    return (np.abs(values - target)).argmin()

def convert_to_indices(radii, elevation, coordinates):
    """Convert physical coordinates to indices in the radii and elevation grids."""
    rad_idx = find_nearest_index(radii, coordinates[0])
    elev_idx = find_nearest_index(elevation, coordinates[1])
    return rad_idx, elev_idx

def point_line_distance(p1, p2, point):
    """Calculate the perpendicular distance from `point` to the line through `p1` and `p2`."""
    num = np.abs((p2[0] - p1[0]) * (p2[1] - point[1]) - (p1[0] - point[0]) * (p2[1] - p1[1]))
    den = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return num / den

def determine_bounds(centers, distance, image_size):
    """Determine bounds for a square around each center within the given image size."""
    bounds = np.array([centers - distance, centers + distance])
    bounds = np.clip(bounds, 0, image_size)
    return bounds

def extract_corners(image):
    """Extract corners from an image using the Shi-Tomasi corner detector."""
    normalized_img = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype('uint8')
    corners = cv2.goodFeaturesToTrack(normalized_img, maxCorners=3, qualityLevel=0.05, minDistance=5, useHarrisDetector=False)
    return np.int0(corners).reshape(-1, 2)

def average_intensity_in_bounds(image, bounds):
    """Calculate average intensity within the specified bounds in the image."""
    intensities = []
    for (start, end) in zip(bounds[0], bounds[1]):
        region = image[start[1]:end[1], start[0]:end[0]]
        intensities.append(calculate_rms(region))
    return intensities

def merge_points(centers, new_centers, corners, corner_intensities, merge_thresh, dist_thresh):
    """Merge points based on distance thresholds and update with corner data if applicable."""
    for i in range(len(new_centers)):
        for j, corner in enumerate(corners):
            if np.linalg.norm(new_centers[i] - corner) < dist_thresh:
                if corner_intensities[j] > corner_intensities[i]:
                    new_centers[i] = corner
                    corner_intensities[i] = corner_intensities[j]
    return new_centers, corner_intensities

def update_image(image, centers, distance, image_size, merge_thresh, dist_thresh):
    """Update frame with new centers if average value is greater than threshold."""
    bounds = determine_bounds(centers, distance, image_size)
    intensities = average_intensity_in_bounds(image, bounds)
    corners = extract_corners(image)
    corner_intensities = average_intensity_in_bounds(image, bounds)
    new_centers, new_intensities = merge_points(centers, centers, corners, corner_intensities, merge_thresh, dist_thresh)
    return new_centers, new_intensities

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