import os
import datetime
import time
import sys
from typing import Literal
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.activations import tanh, elu, relu
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
import h5py

from compare import Compare
from utils import Data, IO

def main(
    source_dir:str,
    training_mode:Literal['fast', 'accurate', 'extended'] = 'fast',
    split_data:int = 1,
    batch_size:int = 4096,
    input_size:int = 100,
    max_epochs:int = 1,
    test_pct:int = 20
):
    """
    A Tensorflow/Keras implementation of the LSTM model found in:
    "Real-Time Guitar Amplifier Emulation with Deep Learning"
    https://www.mdpi.com/2076-3417/10/3/766/htm

    Uses a stack of two 1D Convolutional layers, followed by LSTM,
    followed by a Dense (fully connected) layer. Three preset training
    modes are available, with further customization by editing the code.
    A Sequential tf.keras model is implemented here.

    Note: RAM may be a limiting factor for the parameter "input_size".
    The wav data is preprocessed and stored in RAM, which improves training
    speed but quickly runs out if using a large number for "input_size".
    Reduce this if you are experiencing RAM issues. Use the "--split_data"
    option to divide the data by the specified amount and train the model
    on each set. Doing this will allow for a higher input_size setting
    (and therefore more accurate results).
    
    Modes:
        Speed training (default)
        Accuracy training
        Extended training (set max_epochs as desired, for example 50+)
    """
    print(f"Training started")

    timestamp = datetime.datetime.now().strftime("%Y-%b-%d-%H%M%S").lower()
    name, in_files, out_files = IO.validate_source(source_dir)
    out_dir = f'{name}_{timestamp}'
    os.makedirs(f'models/{out_dir}')
    test_size = test_pct/100

    if training_mode == 'extended':
         # Extended Training (~60x longer than Accuracy Training)
        learning_rate = 0.0005
        conv1d_strides = 3
        conv1d_filters = 36
        hidden_units= 96
    elif training_mode == 'accurate':
        # ~10x longer than Speed Training
        learning_rate = 0.01
        conv1d_strides = 4
        conv1d_filters = 36
        hidden_units= 64
    else:
        learning_rate = 0.01
        conv1d_strides = 12
        conv1d_filters = 16
        hidden_units = 36
    print(f'Params set for training mode "{training_mode}"')

    # =========================================================================
    # Create Sequential Model
    # -------------------------------------------------------------------------
    # A Sqeuential model is a linear stack of layers. "Same" padding is used to
    # ensure the output is the same size as the input. The input shape is the
    # number of samples (input_size) and the number of features (1). The model
    # is compiled with the Adam optimizer and the custom loss function "esr".
    #
    # The history object is used to store the results of model.fit and append
    # metadata about the training process to the saved model file.
    # =========================================================================
    clear_session()
    history = None
    model = Sequential()
    model.add(Input(shape=(input_size, 1)))
    model.add(Conv1D(conv1d_filters, 12, strides=conv1d_strides, activation=None, padding='same'))
    model.add(Conv1D(conv1d_filters, 12, strides=conv1d_strides, activation=None, padding='same'))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation=None))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Data.esr, metrics=[Data.esr])
    print('\n') ;model.summary(); print('\n')

    # =========================================================================
    # Load and Preprocess Data
    # -------------------------------------------------------------------------
    # The input and output wav files are read and stored in numpy arrays. The
    # data is normalized to the maximum value of the input data. The input data
    # is then split into chunks of size "input_size" and the output data is
    # shifted by "input_size-1" to align with the input data. The data is then
    # shuffled and split into training and validation sets for training.
    #
    # NOTE: This currently uses only the first file in each data dir.
    # Future versions likely to require multiple files for training.
    # =========================================================================

    in_sr, in_data = wavfile.read(in_files[0])
    out_sr, out_data = wavfile.read(out_files[0])

    # Validate that sample rates match
    if(in_sr != out_sr):
        # It's pretty trivial to do this conversion, but given that it's
        # a strange situation to be in, preference on raising, as it
        # probably indicates that something weird is going on.
        raise ValueError("Input and output files have different sample rates.")
    else:
        sr = in_sr

    x_all = in_data.astype(np.float32).flatten()
    x_all = Data.normalize(x_all).reshape(len(x_all),1)
    y_all = out_data.astype(np.float32).flatten()
    y_all = Data.normalize(y_all).reshape(len(y_all),1)

    # If splitting the data for training, do this part
    if split_data > 1:
        num_split = len(x_all) // split_data
        x = x_all[0:num_split*split_data]
        y = y_all[0:num_split*split_data]
        x_data = np.split(x, split_data)
        y_data = np.split(y, split_data)

        # Perform training on each split dataset
        for i in range(len(x_data)):
            print(f'Training on split data {(i+1)} of {len(x_data)}...')
            x_split = x_data[i]
            y_split = y_data[i]

            y_ordered = y_split[input_size-1:]

            indices = np.arange(input_size) + np.arange(len(x_split)-input_size+1)[:,np.newaxis]
            x_ordered = tf.gather(x_split,indices)

            shuffled_indices = np.random.permutation(len(x_ordered))
            x_random = tf.gather(x_ordered,shuffled_indices)
            y_random = tf.gather(y_ordered, shuffled_indices)

            # Train Model ###################################################
            split_history = model.fit(x_random,y_random, epochs=max_epochs, batch_size=batch_size, validation_split=test_size)
            if history is not None:
                for key in history.history.keys():
                    history.history[key].extend(split_history.history[key])
            else:
                history = split_history
    # If training on the full set of input data in one run, do this part
    else:
        print(f'Training on unsplit dataset...')
        y_ordered = y_all[input_size-1:]

        indices = np.arange(input_size) + np.arange(len(x_all)-input_size+1)[:,np.newaxis]
        x_ordered = tf.gather(x_all,indices)

        shuffled_indices = np.random.permutation(len(x_ordered))
        x_random = tf.gather(x_ordered,shuffled_indices)
        y_random = tf.gather(y_ordered, shuffled_indices)

        # Train Model ###################################################
        history = model.fit(x_random,y_random, epochs=max_epochs, batch_size=batch_size, validation_split=test_size)

    # Run Prediction #################################################
    print("Running prediction...")

    # Get the last 20% of the wav data to run prediction and plot results
    x_test_data = x_all[-int(len(x_all) * test_size):]
    y_test_data = y_all[-int(len(y_all) * test_size):]

    y_test = y_test_data[input_size-1:]
    indices = np.arange(input_size) + np.arange(len(x_test_data)-input_size+1)[:,np.newaxis]
    x_test = tf.gather(x_test_data,indices)

    prediction = model.predict(x_test, batch_size=batch_size)

    wavfile.write(f'models/{out_dir}/y_pred.wav', sr, prediction.flatten().astype(np.float32))
    wavfile.write(f'models/{out_dir}/x_test.wav', sr, x_test_data.flatten().astype(np.float32))
    wavfile.write(f'models/{out_dir}/y_test.wav', sr, y_test.flatten().astype(np.float32))
    print("Prediction complete \n")

    # =========================================================================
    # Save file
    # -------------------------------------------------------------------------
    # The  model needs to be saved before the additional h5py metadata can
    # be added.
    #
    # Keras complains about saving legacy H5 files, but the h5py metadata
    # append is not forward compatible. This suppresses the warning.
    # Amenable to reviewing h5 metadata usage and dropping if unused or
    # switching to a lib that does equivalent things for keras files.
    # =========================================================================

    model_save_path = f'models/{out_dir}/{name}.h5'

    sys.stderr = open(os.devnull, "w")  # silence stderr
    model.save(model_save_path)
    sys.stderr = sys.__stderr__  # unsilence stderr

    with h5py.File(model_save_path, 'a') as f:
        grp = f.create_group("info")

        grp.attrs['input_size'] = input_size
        grp.attrs['epochs'] = max_epochs
        grp.attrs['batch_size'] = batch_size
        grp.attrs['learning_rate'] = learning_rate
        grp.attrs['model_architecture'] = str(model.to_json())
        grp.attrs['data_preprocessing'] = 'normalized'
        grp.attrs['training_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        grp.attrs['final_training_loss'] = history.history['loss'][-1]
        grp.attrs['final_validation_loss'] = history.history['val_loss'][-1]

    print(f"Training complete. Model saved to: {model_save_path}")
    Compare(out_dir)