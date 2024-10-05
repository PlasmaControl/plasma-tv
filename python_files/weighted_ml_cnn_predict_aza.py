import numpy as np
from pathlib import Path
import h5py
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import EarlyStopping

from src.data.file_utils import GetTV

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

flatten = lambda x : x.reshape(len(x), -1)

def nearest_index(array, value):
    """Find the index of the nearest value in an array."""
    return (np.abs(array - value)).argmin()

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

def reduce_res_2darr(arr, num):
    """Reduce a 2D array by averaging over blocks of size num."""
    return arr[:, ::num, ::num]

run_type = 'all'
prep_filename = 'weighted_outer_dataset_' + run_type
prediction_filename = 'weighted_outer_' + run_type
algorithm = 'linear'
split_ratio = 0.2
files_path = Path('../data/raw/tv_images/aza_9_12').rglob('*.pkl')
files = [f for f in files_path]
ml_id = ''
file_name = prep_filename + '.h5'

mdl2 = load_model(f"/scratch/gpfs/nc1514/plasma-tv/models/weighted_outer_all.keras")
weight_ml_point_save_path = Path('../data/processed/weight_ml_point_aza') / ml_id
for file in files:
    
    frames_vid, _ = pickle.load(open(file, 'rb'),encoding='latin1')
    frames_vid = np.round(frames_vid * (255.0/1000.0)).astype('uint8')[:,::2,:]

    print('Shot:', file.stem.split('_')[-1])
    point_save_name = weight_ml_point_save_path / f"dl{file.stem}.pkl"
    tv_image = frames_vid
    tv_image = reduce_res_2darr(tv_image, 3)
    tv_image = np.expand_dims(tv_image, axis=3)
    
    prediction_cartesian = mdl2.predict(tv_image)
    pickle.dump(prediction_cartesian, open(point_save_name, 'wb'))
    print('Saved:', point_save_name)