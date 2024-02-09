import h5py
import numpy as np
import torch
from pathlib import Path
from data.base_dataset import BaseDataset


class HDF5Dataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.AB_path = Path(opt.dataroot)  # get the image directory
        self.input_nc = (
            self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        )
        self.output_nc = (
            self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc
        )
        with h5py.File(self.AB_path, "r") as f:
            print(list(f.keys()))
            self.A_arr = f["A_" + opt.phase][:]
            self.B_arr = f["B_" + opt.phase][:]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """

        # read a image given a random integer index
        A = self.A_arr[index]
        B = self.B_arr[index]

        A = np.expand_dims(A, axis=0)
        B = np.expand_dims(B, axis=0)

        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        return {
            "A": A,
            "B": B,
            "A_paths": str(self.AB_path.stem) + "_A_" + str(index),
            "B_paths": str(self.AB_path.stem) + "_B_" + str(index),
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_arr)
