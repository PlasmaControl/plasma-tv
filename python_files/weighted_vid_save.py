import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import pickle
from tqdm.notebook import tqdm

from src.data.file_utils import GetTV

# scp -r -o 'ProxyCommand ssh -p 2039 chenn@cybele.gat.com -W %h:%p' chenn@iris.gat.com:/cscratch/jalalvanda/tangtv .

ml_id = ''
run_type = 'aza'

prep_filename = 'weighted_outer_dataset_' + run_type
prediction_filename = 'weighted_outer_' + run_type
algorithm = 'linear'
split_ratio = 0.2

tv_path = Path('../data/raw/tv_images') / run_type
label_path = Path('../data/labels/weighted_emission') / run_type
prep_path = Path('../data/processed/hdf5')
model_path = Path('../models')

mp4_save_path = Path('../outputs/video/weighted_dl') / run_type
mp4_save_path.mkdir(parents=True, exist_ok=True)

weight_ml_point_save_path = Path('../data/processed/weight_ml_point') / ml_id

tv = GetTV(tv_path)
files = tv.list_files()
file_lengths = tv.file_lengths()
cumulative_lengths = np.insert(np.cumsum(file_lengths), 0, 0)
tv_dim = tv.load(files[0], 'vid').shape

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

for file in files:
    
    print('Shot:', file.stem.split('_')[-1])
    mp4_save_name = mp4_save_path / f"{file.stem.split('_')[-1]}.mp4"
    label_file = (label_path / file.stem).with_suffix('.pkl')
    point_save_name = weight_ml_point_save_path / f"dl{file.stem}.pkl"
    with open(label_file, 'rb') as f:
        labels_cartesian = pickle.load(f)
    frames = tv.load(file, 'frames').astype('int')
    tv_image = tv.load(file, 'vid')[frames]
    vid_input = np.expand_dims(tv_image, axis=3)
    inverted = tv.load(file, 'inverted')
    elevation = tv.load(file, 'elevation')[0]
    radii = tv.load(file, 'radii')[0]
    with open(point_save_name, 'rb') as f:
        prediction_cartesian = pickle.load(f)
    prediction = get_index(prediction_cartesian, elevation, )
    labels = get_index(labels_cartesian, elevation)

    # Initialize figure and axes
    print("Animating...")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    img = ax.pcolor(radii, elevation, inverted[0])
    hline_label = ax.axhline(labels_cartesian[0], c='lime', label='label')
    hline_prediction = ax.axhline(prediction_cartesian[0], c='red', label='prediction', ls='--')
    ax.legend(loc='upper right')
    ax.set_title(f'Inverted View: 0')
    fig.suptitle(f"Shot {file.stem.split('_')[-1]}")
    frames = []
    # Function to update the plot
    def update(idx):
        img.set_array(inverted[idx])
        
        hline_label.set_ydata([labels_cartesian[idx]])
        hline_prediction.set_ydata([prediction_cartesian[idx]])
        
        ax.set_title(f'Inverted View: {idx}')
        
        return img, hline_label, hline_prediction
        
    # Create the animation using FuncAnimation
    ani = animation.FuncAnimation(fig, update, frames=range(inverted.shape[0]), blit=True, repeat=False)

    # Save the animation as an MP4 file
    print("Saving MP4...")
    FFwriter = animation.FFMpegWriter(fps=30, extra_args=["-vcodec", "libx264"])
    ani.save(mp4_save_name, writer=FFwriter)

    plt.close(fig)