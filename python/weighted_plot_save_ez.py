import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

flatten = lambda x : x.reshape(len(x), -1)

# set this #
mdl_path = '../models/weighted_outer_all_coefficients_noisify.txt'
intercept = -1.1618299423292409
vid_path = '../data/raw/tv_images/h-mode'
sav_or_pkl = 'sav' # fillipo files in sav, aza_files in pkl
# -------- #
with open(mdl_path, "r") as file:
    coef = file.readlines()
z_coef_ = np.array([float(i) for i in coef])
inter_ = intercept

if sav_or_pkl == 'sav':
    files_path = Path(vid_path).rglob('*.sav')
elif sav_or_pkl == 'pkl':
    files_path = Path(vid_path).rglob('*.pkl')
files = [f for f in files_path]
if files == []:
    raise ValueError("No files found")

for file in files:
        print(f"Processing {file}")
        if sav_or_pkl == 'sav':
            print('hi')
        elif sav_or_pkl == 'pkl':
        frames_vid, times_vid = pickle.load(open(file, 'rb'),encoding='latin1')
        frames_vid = np.round(frames_vid * .125).astype('uint8')[:,::2,:]
        frames_vid_flatten = flatten(frames_vid)
        prediction_cartesian_regression = [np.dot(z_coef_, frame) + inter_ for frame in frames_vid_flatten]
        # prediction_cartesian_regression = model.predict(frames_vid_flatten)
        plt.plot(times_vid, prediction_cartesian_regression)
        plt.xlabel('Time (s)')
        plt.ylabel('Predicted Elevation (m)')
        plt.title('Shot: ' + file.stem.split('_')[-1])
        plt.savefig(f"../outputs/plots/aza_9_18/{file.stem}_elevation.png")
        plt.close()
    
        txt_file = (Path('../data/processed/weight_ml_point/txt_files') / file.stem.split('_')[-1]).with_suffix('.txt')
        with open(txt_file, 'w') as f:
                for idx in range(len(prediction_cartesian_regression)):
                        f.write(f"{times_vid[idx]}, {prediction_cartesian_regression[idx]}\n")
        print(f"Saved {txt_file}")
