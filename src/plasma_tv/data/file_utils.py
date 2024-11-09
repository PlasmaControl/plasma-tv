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

class GetPkl:
    def __init__(self, file_path="data/raw/"):
        self.file_path = Path(file_path).rglob('*.pkl')
    
    def list_files(self, display=False):
        files = sorted([f for f in self.file_path])
        if display:
            for idx, file in enumerate(files):
                print(idx, '\t',file.stem.split('_')[-1])
        else:
            print('Number of files:', len(files))
        return files
    
    def load_raw(self, file_path):
        return  pickle.load(open(file_path, 'rb'),encoding='latin1')
    
    def load_processed(self, file_path):
        vid, time = self.load_raw(file_path)
        return  np.round(vid * (255.0/1000.0)).astype('uint8')[:,::2,:], time
    
class GetEmission:
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

    def __init__(self, file_path="data/raw/", file_key="emission_structure"):
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

    def list_files(self, display=False):
        """
        Returns a list of files in the file path.

        Returns:
            list: A list of files in the file path.
        """
        files = sorted([f for f in self.file_path.iterdir() if (f.suffix == ".sav")])
        
        if display:
            for idx, file in enumerate(files):
                print(idx, '\t',file.stem.split('_')[-1])
        else:
            print('Number of files:', len(files))
            
        return files

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

    def load_all(self, file_path, resize=True):
        """
        Loads all TV data types from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            list: A list of loaded TV data types.
        """
        print('Extracting sav for shot:', file_path.stem.split('_')[-1])
        [inverted,radii,elevation,frames,times,vid_frames,vid_times,vid] = [readsav(file_path)[self.file_key][0][type] for type in range(8)]
        if resize:
            inverted_dim = inverted.shape
            if (inverted_dim[1] != 201) or (inverted_dim[2] != 201):
                print('Resizing...')
                inverted = inverted[:,:201,:201]
                radii = radii[:,:201]
                elevation = elevation[:,:201]
                inverted_dim = inverted.shape
        return [inverted,radii,elevation,frames,times,vid_frames,vid_times,vid]
