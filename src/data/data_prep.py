import numpy as np
from file_utils import save_to_hdf5, load_pickle
from utils.get_file import GetTV
from pathlib import Path

def prepare_and_save_data(files, tv, data_path, temp_hdf5_path):
    file_lengths = [tv.file_len(f, False) for f in files]
    cumulative_lengths = np.insert(np.cumsum(file_lengths), 0, 0)
    
    for idx, file in enumerate(files):
        tv_image = np.asarray(tv._load_data(file, 'vid'))
        label_path = data_path / f"{file.stem}.pkl"
        labels = load_pickle(label_path)
        
        points = np.concatenate((labels['l_location'], labels['r_location']), axis=1)
        
        dataset_path = temp_hdf5_path / f"{file.stem}_data.hdf5"
        save_to_hdf5(tv_image, dataset_path, "tv_images")
        save_to_hdf5(points, dataset_path, "points", mode='a')

    return cumulative_lengths  # Optionally return metadata for further use