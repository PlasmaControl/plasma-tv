import numpy as np
from pathlib import Path
from scipy.io import readsav
import pickle
import h5py

# [inverted,radii,elevation,frames,times,vid_frames,vid_times,vid] = _load_data(filepath / fullfile)

def save_hdf5(data, file_name, dataset_name, data_shape, data_type):
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset(dataset_name, shape=data_shape, dtype=data_type, data=data)

def load_pickle(file_path):
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)

def write_pickle(data, file_path):
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
        
class H5Parse:
    """
    A class for reading and writing data from an HDF5 file.
    """

    def read(self, filepath):
        """
        Read data from the specified HDF5 file.

        Args:
            filepath (str): The path to the HDF5 file.

        Returns:
            dict: A dictionary containing the data read from the file.
        """
        with h5py.File(filepath, "r") as f:
            data = {key: f[key][:] for key in f}
        return data

    def list_keys(self, filepath):
        """
        List all the keys in the specified HDF5 file.

        Args:
            filepath (str): The path to the HDF5 file.

        Returns:
            list: A list of keys present in the file.
        """
        with h5py.File(filepath, "r") as f:
            return [key for key in f]

    def write(self, filepath, data):
        """
        Write data to the specified HDF5 file.

        Args:
            filepath (str): The path to the HDF5 file.
            data (dict): A dictionary containing the data to be written.

        Returns:
            None
        """
        with h5py.File(filepath, "w") as f:
            for key in data:
                f.create_dataset(key, data[key])


class GetTV:
    """
    A class for handling TV files.

    Attributes:
        file_path (Path): The path to the TV files.
        file_key (str): The key used to access the TV data in the files.
        index_dict (dict): A dictionary mapping TV data types to their indices.

    Methods:
        __init__(self, file_path='tv_images', file_key='emission_structure'): Initializes a new instance of the GetTV class.
        __len__(self): Returns the number of files in the file path.
        change_file_path(self, file_path): Changes the file path.
        list_files(self): Returns a list of files in the file path.
        file_len(self, file_path, inversion=True): Returns the length of a specific TV data type in a file.
        load(self, file_path, type): Loads a specific TV data type from a file.
        load_all(self, file_path): Loads all TV data types from a file.
    """

    def __init__(self, file_path="data/raw/tv_images", file_key="emission_structure"):
        """
        Initializes a new instance of the GetTV class.

        Args:
            file_path (str): The path to the TV files.
            file_key (str): The key used to access the TV data in the files.
        """
        self.file_path = Path(file_path)
        self.file_key = file_key
        self.index_dict = {
            "inverted": 0,
            "radii": 1,
            "elevation": 2,
            "frames": 3,
            "times": 4,
            "vid_frames": 5,
            "vid_times": 6,
            "vid": 7,
        }

    def __len__(self):
        """
        Returns the number of files in the file path.

        Returns:
            int: The number of files in the file path.
        """
        return len(self._list_files())

    def change_file_path(self, file_path):
        """
        Changes the file path.

        Args:
            file_path (str): The new file path.
        """
        self.file_path = Path(file_path)

    def list_files(self):
        """
        Returns a list of files in the file path.

        Returns:
            list: A list of files in the file path.
        """
        return sorted([f for f in self.file_path.iterdir() if (f.suffix == ".sav")])

    def file_len(self, file_path, inversion=True):
        """
        Returns the length of a specific TV data type in a file.

        Args:
            file_path (str): The path to the file.
            inversion (bool): Whether to consider the inverted TV data type. Default is True.

        Returns:
            int: The length of the TV data type in the file.
        """
        if inversion:
            return len(readsav(file_path)[self.file_key][0][3])
        else:
            return len(readsav(file_path)[self.file_key][0][5])
        
    def file_lengths(self):
        file_lengths = [self.file_len(f, False) for f in self.list_files()]
        return file_lengths

    def load(self, file_path, type):
        """
        Loads a specific TV data type from a file.

        Args:
            file_path (str): The path to the file.
            type (str): The TV data type to load.

        Returns:
            object: The loaded TV data.
        """
        dat = readsav(file_path)[self.file_key][0][self.index_dict[type]]
        return dat

    def load_all(self, file_path):
        """
        Loads all TV data types from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            list: A list of loaded TV data types.
        """
        dat = [readsav(file_path)[self.file_key][0][type] for type in range(8)]
        return dat
