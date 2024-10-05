import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from pathlib import Path
import pickle
from tqdm.notebook import tqdm
from scipy.interpolate import interp1d
import h5py
from src.data.file_utils import GetTV

# functions
flatten = lambda x : x.reshape(len(x), -1)

def nearest_index(array, value):
    return (np.abs(array - value)).argmin()

def crop_time(times, data, start_time, end_time):
    if data.ndim == 1:
        data = data.reshape(1, -1)
    start_idx = nearest_index(times, start_time)
    end_idx = nearest_index(times, end_time)
    return times[start_idx:end_idx], data[:,start_idx:end_idx]

def get_index(arr, coord):
    ind = np.searchsorted(coord, arr)
    ind = np.clip(ind, 0, len(coord) - 1)
    for i, cval in enumerate(arr):
        if ind[i] > 0 and abs(cval - coord[ind[i] - 1]) < abs(cval - coord[ind[i]]):
            ind[i] -= 1
    return ind

input_dir = 'all'
output_id = 'noisy26' # output will be in this directory ('../output/input_dir/output_id')
sav_or_pkl = 'sav' # sav is fillipo, pkl is aza
model_name = '092624_tangtv_v5.txt' # name of the model file coefficients
interp_kind = 'next' # method of interpolating input and mds data
interp_points = 1000 # number of points to interpolate to
intercept = -0.906314

input_path = Path('../data/raw/tv_images') / input_dir
output_path = Path('../outputs')
label_path = Path('../data/labels/weighted_emission') / input_dir
model_path = Path('../models')
h5_path = '../data/external/toksearch/detach.h5' # use detach for all, aza_2 for aza

# path to save mp4 video and text files
plot_save_path = output_path / 'plot' / input_dir / output_id
plot_save_path.mkdir(parents=True, exist_ok=True)
value_save_path = output_path / 'value' / input_dir / output_id
value_save_path.mkdir(parents=True, exist_ok=True)

# input path
if sav_or_pkl == 'sav':
    print(f"Loading Inupts from {input_path}")
    tv = GetTV(input_path)
    files = tv.list_files()
elif sav_or_pkl == 'pkl':
    files = sorted(list(input_path.rglob('*.pkl')))

with open(model_path / model_name, "r") as file:
    coef = file.readlines()
z_coef_ = np.array([float(i) for i in coef])
inter_ = intercept

files = files[-12:]
for file in files:
    print('Shot:', file.stem.split('_')[-1])
    
    # load input data
    if sav_or_pkl == 'sav':
        input = tv.load(file, 'vid')
        input_time = tv.load(file, 'vid_times')
    elif sav_or_pkl == 'pkl':
        frames_vid, times_vid = pickle.load(open(file, 'rb'),encoding='latin1')
        input = np.round(frames_vid * .25).astype('uint8')[:,::2,:]
        input_time = times_vid * 1000
    input_flatten = flatten(input)
    data = np.array([np.dot(z_coef_, frame) + inter_ for frame in input_flatten])
    
    # load zxpt and flattop times
    shotnum = str(file.stem.split('_')[-1])
    with h5py.File(h5_path, 'r') as h5file:
        h5_times = h5file['times'][:]
        t_ip_flat_sql = h5file[shotnum]['t_ip_flat_sql'][()]
        ip_flat_duration_sql = h5file[shotnum]['ip_flat_duration_sql'][()]
        if (t_ip_flat_sql == 'nan') or (ip_flat_duration_sql == 'nan'):
            print(f"Shot {shotnum} has no flattop data")
            continue
        if 'ZXPT1_EFIT01' not in h5file[shotnum]:
            print(f"Shot {shotnum} has no ZXPT1_EFIT01 data")
            continue
        ZXPT1_EFIT01 = h5file[shotnum]['ZXPT1_EFIT01'][:]
    t_start = t_ip_flat_sql
    t_end = t_ip_flat_sql + ip_flat_duration_sql

    # crop and interpolate times
    t_out = np.linspace(t_start, t_end, interp_points)
    crop_h5_times, crop_ZXPT1_EFIT01 = crop_time(h5_times, ZXPT1_EFIT01, t_start, t_end)
    crop_times, crop_data = crop_time(input_time, data, t_start, t_end)
    if (len(crop_times) == 0) or (len(crop_h5_times) == 0):
        if len(crop_times) == 0:
            print(f"Shot {shotnum} has no video data in the flattop")
        if len(crop_h5_times) == 0:
            print(f"Shot {shotnum} has no zxpt data in the flattop")
        continue

    print(f"Interpolating between {t_start} and {t_end}")
    interp_ZXPT = interp1d(crop_h5_times, crop_ZXPT1_EFIT01, kind=interp_kind, fill_value='extrapolate')(t_out)[0]
    interp_data = interp1d(crop_times, crop_data, kind=interp_kind, fill_value='extrapolate')(t_out)[0]
    # set strike point to -1.24477
    interp_ZStrike = np.ones_like(interp_ZXPT) * -1.24477
    
    # calculate Dz
    denominator = interp_ZXPT - interp_ZStrike
    Dz = (interp_ZXPT - interp_data) / denominator
    Dz_out = Dz
    if np.any(denominator == 0):
        print(f"Shot {shotnum} has denominator of 0 in Dz. Setting Nans to 0")
        Dz_out[np.where(denominator < 1e-2)] = 0
    
    # plot Dz
    save_file_path = plot_save_path / f"{file.stem}_dz.png"
    time_range = [1000, 5000]
    idx_range = (nearest_index(t_out, time_range[0]), nearest_index(t_out, time_range[1]))
    Dz_range = np.array([np.min(Dz_out[idx_range[0]:idx_range[1]]), np.max(Dz_out[idx_range[0]:idx_range[1]])])
    if np.isnan(Dz_range).any() or np.isinf(Dz_range).any():
        print(f"Shot {shotnum} has NaN or Inf in Dz range. Setting range to -2,2")
        Dz_range = [-1, 1]
    plt.plot(t_out, Dz_out)
    plt.xlim(time_range[0],time_range[1])
    plt.ylim(Dz_range[0],Dz_range[1])
    plt.xlabel('Time (ms)')
    plt.ylabel('Dz')
    plt.title('Shot: ' + file.stem.split('_')[-1])
    plt.savefig(save_file_path)
    plt.close()
    print(f"Saved {save_file_path}")
    
    # save Dz to txt file
    value_file_path = value_save_path / f"{file.stem}_dz.txt"
    with open(value_file_path, 'w') as f:
        for idx in range(len(Dz_out)):
            f.write(f"{t_out[idx]}, {Dz_out[idx]}\n")
    print(f"Saved {value_file_path}")