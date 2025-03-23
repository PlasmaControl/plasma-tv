from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import h5py

from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from plasma_tv.data.file_utils import GetEmission
from plasma_tv.models import predict #model
from plasma_tv.utils.helpers import *

models = predict.list_models()

run_type = 'all'
prep_filename = 'weighted_outer_dataset_reg_' + run_type
prediction_filename = 'weighted_outer_' + run_type
algorithm = 'LinearRegression'
split_ratio = 0.2
tv_path = Path('../data/raw') / run_type
label_path = Path('../data/labels/weighted_emission') / run_type
prep_path = Path('../data/processed/hdf5')
model_path = Path('../models')
xpt_path = Path('../data/external/toksearch/12_03_2024_bfield.h5')
tv = GetEmission(tv_path)
files = tv.list_files()
file_lengths = tv.file_lengths()
cumulative_lengths = np.insert(np.cumsum(file_lengths), 0, 0)
tv_dim = tv.load(files[0], 'vid').shape
ml_id = '6'
file_name = prep_filename + '.h5'
target_cdf = np.load('../data/processed/histogram/target_cdf.npy')
bin_edges = np.load('../data/processed/histogram/bin_edges.npy')

file_name = prep_filename + '.h5'

with h5py.File(prep_path / file_name, 'r') as f:
    vid_train = f['vid_train'][:]
    points_train = f['points_train'][:]
    rxpt_train = f['rxpt_train'][:]
    zxpt_train = f['zxpt_train'][:]
    vid_test = f['vid_test'][:]
    points_test = f['points_test'][:]
    rxpt_test = f['rxpt_test'][:]
    zxpt_test = f['zxpt_test'][:]

files = tv.list_files()
elevation = tv.load(files[0], 'elevation')[0]
radii = tv.load(files[0], 'radii')[0]
vid_shape = tv.load(files[0], 'vid')[0].shape

X_train = flatten(vid_train)
train_add = rxpt_train[:, np.newaxis]
X_train = np.concatenate([X_train,train_add], axis=1)
X_test = flatten(vid_test)
test_add = rxpt_test[:, np.newaxis]
X_test = np.concatenate([X_test,test_add], axis=1)

y_train = points_train
y_test = points_test

train_nan_indices = np.where(np.isnan(train_add))[0]
train_inf_indices = np.where(np.isinf(train_add))[0]
test_nan_indices = np.where(np.isnan(test_add))[0]

print('Train NaN Indices:', train_nan_indices)
print('Test NaN Indices:', test_nan_indices)

X_train = np.delete(X_train, train_nan_indices, axis=0)
y_train = np.delete(y_train, train_nan_indices, axis=0)
X_test = np.delete(X_test, test_nan_indices, axis=0)
y_test = np.delete(y_test, test_nan_indices, axis=0)

mdl = LinearRegression()
mdl.fit(X_train, y_train)

r_predict = mdl.predict(X_test)
err = mean_absolute_error(r_predict,y_test) * 100
print(f'Mean Absolute Error: {err}')
print('Root Mean Squared Error: ', root_mean_squared_error(r_predict, y_test)*100)

coefficients = mdl.coef_
intercept = mdl.intercept_

coefficients_file = model_path / f"{prediction_filename}{ml_id}_coefficients.txt"
with open(coefficients_file, 'w') as f:
    for coef in coefficients:
        f.write(f"{coef}\n")
print(coefficients_file, "has been saved!")
print("Intercept (Please Write Down):", intercept)