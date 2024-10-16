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

run_type = 'campaign_24'
prep_filename = 'weighted_outer_dataset_' + run_type
prediction_filename = 'weighted_outer_' + run_type
algorithm = 'linear'
split_ratio = 0.2
tv_path = Path('../data/raw/tv_images') / run_type
label_path = Path('../data/labels/weighted_emission') / run_type
prep_path = Path('../data/processed/hdf5')
model_path = Path('../models')
tv = GetTV(tv_path)
files = tv.list_files()
file_lengths = tv.file_lengths()
cumulative_lengths = np.insert(np.cumsum(file_lengths), 0, 0)
tv_dim = tv.load(files[0], 'vid').shape
ml_id = ''
file_name = prep_filename + '.h5'

mdl2 = load_model(f"{model_path / prediction_filename}{ml_id}.keras")
weight_ml_point_save_path = Path('../data/processed/weight_ml_point/dl') / ml_id
for file in files:
    
    print('Shot:', file.stem.split('_')[-1])
    point_save_name = weight_ml_point_save_path / f"dl{file.stem}.pkl"
    label_file = (label_path / file.stem).with_suffix('.pkl')
    with open(label_file, 'rb') as f:
        labels_cartesian = pickle.load(f)
    tv_image = tv.load(file, 'vid')
    tv_image = reduce_res_2darr(tv_image, 3)
    tv_image = np.expand_dims(tv_image, axis=3)
    
    prediction_cartesian = mdl2.predict(tv_image)
    pickle.dump(prediction_cartesian, open(point_save_name, 'wb'))