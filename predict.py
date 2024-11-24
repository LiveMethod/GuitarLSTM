import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

from utils import Data

def predict(args):
    '''
    Predicts the output wav given an input wav file, trained GuitarLSTM model, 
    and output wav filename.
    '''
    # Read the input_size from the .h5 model file
    f = h5py.File(args.model, 'a')
    input_size = f["info"]["input_size"][0]
    f.close()

    # Load model from .h5 model file
    name = args.out_filename
    model = load_model(args.model, custom_objects={'error_to_signal' : Data.esr})
    
    # Load and Preprocess Data
    print("Processing input wav")
    sr, in_data = wavfile.read(args.in_file)

    X = in_data.astype(np.float32).flatten()
    X = Data.normalize(X).reshape(len(X),1)

    indices = np.arange(input_size) + np.arange(len(X)-input_size+1)[:,np.newaxis] 
    X_ordered = tf.gather(X,indices) 

    # Run prediction and save output audio as a wav file
    print("Running prediction")
    prediction = model.predict(X_ordered, batch_size=args.batch_size)
    print("Prediction success")
    out_file = name
    if not name.endswith('.wav'):
        out_file = name + '.wav'
    wavfile.write(name, sr, prediction.flatten().astype(np.float32))
    print(f'Prediction saved to: {out_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_filename")
    parser.add_argument("model")
    parser.add_argument("--train_data", default="data.pickle")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    predict(args)