import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import EarlyStopping

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

run_type = 'all'
prep_filename = 'weighted_outer_dataset_' + run_type
prep_path = Path('../data/processed/hdf5')
model_path = Path('../models')
prediction_filename = 'weighted_outer_' + run_type

ml_id = ''
file_name = prep_filename + '.h5'

with h5py.File(prep_path / file_name, 'r') as f:
    vid_train = f['vid_train'][:]
    points_train = f['points_train'][:]
    vid_test = f['vid_test'][:]
    points_test = f['points_test'][:]

X_train = np.expand_dims(vid_train, axis=3)
y_train = points_train
X_test = np.expand_dims(vid_test, axis=3)
y_test = points_test

# Define the model
mdl2 = Sequential([
    InputLayer((80, 240, 1)),  # Input shape for grayscale image
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),  # Second Conv layer with ReLU
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),   # Max pooling layer with 2x2 pool size 
    GlobalAveragePooling2D(),# Flatten the output
    Dense(128, activation='relu'),          # First fully connected layer with ReLU                         # Another Dropout layer
    Dense(1, activation='linear')                                # Second fully connected layer (output layer)
])

# Compile the model using Adam optimizer and Mean Squared Error loss
mdl2.compile(optimizer=Adam(learning_rate=0.001),
              loss='mae', 
              metrics=['mse','mae'])
print(mdl2.summary())

# Train the model
callback = EarlyStopping(monitor='loss', patience=6)
history = mdl2.fit(
    X_train, y_train,
    validation_split = 0.05,
    epochs=150,
    batch_size = 5,
    callbacks = [callback],
)

run_type = 'campaign_24'
prep_filename = 'weighted_outer_dataset_' + run_type
prep_path = Path('../data/processed/hdf5')
model_path = Path('../models')
prediction_filename = 'weighted_outer_' + run_type

ml_id = ''
file_name = prep_filename + '.h5'

with h5py.File(prep_path / file_name, 'r') as f:
    vid_train = f['vid_train'][:]
    points_train = f['points_train'][:]
    vid_test = f['vid_test'][:]
    points_test = f['points_test'][:]

X_train = np.expand_dims(vid_train, axis=3)
y_train = points_train
X_test = np.expand_dims(vid_test, axis=3)
y_test = points_test

# Train the model
callback = EarlyStopping(monitor='loss', patience=4)
history2 = mdl2.fit(
    X_train, y_train,
    validation_split = 0.05,
    epochs=20,
    batch_size = 2,
    callbacks = [callback],
)

# Evaluate the model on the test set
test_loss, test_mse, test_mae = mdl2.evaluate(X_test, y_test)

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='val')
plt.legend()
plt.tight_layout()
plt.savefig(f"../outputs/cnnhistory.png")

print(f'Test loss (RMSE): {test_loss}')
print(f'Test RMSE: {np.sqrt(test_mse)}')
print(f'Test MAE: {test_mae}')

mdl2.save(f"{model_path / prediction_filename}{ml_id}.keras")