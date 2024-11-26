import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from utils import Data

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
from utils import Data
from datetime import datetime

input_size = 100
# Load the model
model_path = str('models/005006/model.keras')
wav_path = 'models/005006/x_test.wav'
out_path = 'models/005006/new_prediction.wav'

model = load_model(model_path, custom_objects={'dual_loss' : Data.dual_loss})

# Load the wav file
sr, data = wavfile.read(wav_path)
data = data.astype(np.float32).flatten()
data = Data.normalize(data).reshape(len(data), 1)
gain_data = np.full(data.shape[0], [0.5])

indices = np.arange(input_size) + np.arange(len(data)-input_size)[:,np.newaxis]
c = tf.gather(gain_data, indices)
x = tf.gather(data, indices)

prediction = model.predict({'audio_input': x, 'control_input': c}, batch_size=4096*2)
wavfile.write(out_path, sr, prediction.flatten().astype(np.float32))
