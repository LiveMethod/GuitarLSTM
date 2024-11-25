import os
import datetime
import time
import sys
from typing import Literal
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from keras import Sequential
from keras._tf_keras.keras.layers import LSTM, Conv1D, Dense, Input, Concatenate
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.backend import clear_session
from keras._tf_keras.keras.models import Model
import json

from compare import Compare
from utils import Data, IO

def main(
    training_mode:Literal['fast', 'accurate', 'extended'] = 'fast',
    split_data:int = 1,
    batch_size:int = 4096,
    input_size:int = 100,
    max_epochs:int = 1,
):

    timestamp = datetime.datetime.now().strftime("%H%M%S").lower()
    name="testing_conditioning"
    out_dir = f'{name}_{timestamp}'
    os.makedirs(f'models/{out_dir}')
    print(f"Current output directory: {out_dir}")

    sources = [
      [[0.0], "data/silver_conditioned/c_gain_00.wav", "data/silver_conditioned/raw.wav"],
      [[0.1], "data/silver_conditioned/c_gain_01.wav", "data/silver_conditioned/raw.wav"],
      [[0.2], "data/silver_conditioned/c_gain_02.wav", "data/silver_conditioned/raw.wav"],
      [[0.3], "data/silver_conditioned/c_gain_03.wav", "data/silver_conditioned/raw.wav"],
      [[0.4], "data/silver_conditioned/c_gain_04.wav", "data/silver_conditioned/raw.wav"],
      [[0.5], "data/silver_conditioned/c_gain_05.wav", "data/silver_conditioned/raw.wav"],
      [[0.6], "data/silver_conditioned/c_gain_06.wav", "data/silver_conditioned/raw.wav"],
      [[0.7], "data/silver_conditioned/c_gain_07.wav", "data/silver_conditioned/raw.wav"],
      [[0.8], "data/silver_conditioned/c_gain_08.wav", "data/silver_conditioned/raw.wav"],
      [[0.9], "data/silver_conditioned/c_gain_09.wav", "data/silver_conditioned/raw.wav"],
      [[1.0], "data/silver_conditioned/c_gain_10.wav", "data/silver_conditioned/raw.wav"]
    ]
    for source in sources:
      for path in source[1:]:  # Skip the non-path element
        if not os.path.exists(path):
          raise FileNotFoundError(f"Error: Required audio file not found: {path}")

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    
    conditioning_features = 1 # just gain for now

    if training_mode == 'extended':
         # Extended Training (~60x longer than Accuracy Training)
        learning_rate = 0.0005
        conv1d_strides = 3
        conv1d_filters = 36
        hidden_units= 96
        kernel_size = 12
    elif training_mode == 'accurate':
        # ~10x longer than Speed Training
        learning_rate = 0.01
        conv1d_strides = 4
        conv1d_filters = 36
        hidden_units= 64
        kernel_size = 12
    else:
        learning_rate = 0.01
        conv1d_strides = 12
        conv1d_filters = 8
        hidden_units = 96
        kernel_size = 12
    
    config_summary = {
      "sources": sources,
      "training_mode": training_mode,
      "split_data": split_data,
      "batch_size": batch_size,
      "input_size": input_size,
      "max_epochs": max_epochs,
      "learning_rate": learning_rate,
      "conv1d_strides": conv1d_strides,
      "conv1d_filters": conv1d_filters,
      "hidden_units": hidden_units,
      "kernel_size": kernel_size,
    }

    with open(f'models/{out_dir}/summary.json', 'w') as f:
      json.dump(config_summary, f, indent=2)
    print(f'Params saved for training mode "{training_mode}"')

    # =========================================================================
    # MODEL ARCHITECTURE
    # -------------------------------------------------------------------------
    # =========================================================================
  
    clear_session()
    history = None
    # model = Sequential()

    # Audio processing branch
    audio_input = Input(shape=(input_size, 1), name='audio_input')
    x = Conv1D(conv1d_filters, kernel_size, strides=conv1d_strides, activation=None, padding='same')(audio_input)
    x = Conv1D(conv1d_filters, kernel_size, strides=conv1d_strides, activation=None, padding='same')(x)
    x = LSTM(hidden_units)(x)

    # Control processing branch
    control_input = Input(shape=(conditioning_features,), name='control_input')
    c = Dense(16)(control_input)
    c = Dense(16)(c)
    
    # Merge branches
    merged = Concatenate()([x, c])
    output = Dense(1, activation=None)(merged)

    ###
    model = Model(inputs=[audio_input, control_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Data.dual_loss, metrics=[Data.dual_loss])

    print('\n') ;model.summary(); print('\n')

    # =========================================================================
    # DATA INGESTION
    # =========================================================================

    sr = 44100 # we gon take on some technical debt while I refactor lol

    # =========================================================================
    # TRAINING (SPLIT AND UNSPLIT)
    # =========================================================================

    # Training stage
    history = None
    print(f'Training...')
    validation_c = []
    validation_x = []
    validation_y = []
    

    for batch_num, (train_batch, test_batch) in enumerate(IO.load_conditioned_batch(sources)):
        c_batch, x_batch, y_batch = train_batch
        print(f'Processing batch {batch_num + 1}...')

        # Prepare data
        y_ordered = y_batch[input_size-1:]
        indices = np.arange(input_size) + np.arange(len(x_batch)-input_size+1)[:,np.newaxis]
        x_ordered = tf.gather(x_batch, indices)
        c_ordered = tf.gather(c_batch, indices)
        
        # Shuffle data
        shuffled_indices = np.random.permutation(len(x_ordered))
        c_random = tf.gather(c_ordered, shuffled_indices)
        x_random = tf.gather(x_ordered, shuffled_indices)
        y_random = tf.gather(y_ordered, shuffled_indices)
        
        # Train on batch
        batch_history = model.fit(
            [x_random, c_random], 
            y_random, 
            epochs=max_epochs, 
            batch_size=batch_size, 
            validation_split=0.15
        )

        # Aggregate history
        if history is None:
            history = batch_history
        else:
            for key in history.history.keys():
                history.history[key].extend(batch_history.history[key])
        
        vc, vx, vy = test_batch
    
        # Store validation components
        if validation_x is None:
            validation_c = vc
            validation_x = vx
            validation_y = vy
        else:
            validation_c = np.concatenate([validation_c, vc])
            validation_x = np.concatenate([validation_x, vx])
            validation_y = np.concatenate([validation_y, vy])
            
    
    # =========================================================================
    # PREDICTION
    # =========================================================================

    # Run Prediction #################################################
    print("Running prediction...")
    validation_data = (validation_x, validation_y, validation_c)

    y_test = validation_y[input_size-1:]
    indices = np.arange(input_size) + np.arange(len(validation_x)-input_size+1)[:,np.newaxis]
    x_test = tf.gather(validation_x,indices)
    c_test = tf.gather(validation_c,indices)

    prediction = model.predict([x_test, c_test], batch_size=batch_size)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    wavfile.write(f'models/{out_dir}/y_pred.wav', sr, prediction.flatten().astype(np.float32))
    wavfile.write(f'models/{out_dir}/x_test.wav', sr, x_test.flatten().astype(np.float32))
    wavfile.write(f'models/{out_dir}/y_test.wav', sr, y_test.flatten().astype(np.float32))
    print("Prediction complete \n")

    model_save_path = f'models/{out_dir}/{name}.h5'

    sys.stderr = open(os.devnull, "w")  # silence stderr
    model.save(model_save_path)
    sys.stderr = sys.__stderr__  # unsilence stderr

    model_architecture = json.loads(model.to_json())
    final_summary = {
      **config_summary,
      "data_preprocessing": "normalized",
      "training_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
      "history": history.history,
      "model_architecture": model_architecture
    }

    # Ovbrwrite the config file with additional information
    with open(f'models/{out_dir}/summary.json', 'w') as f:
      json.dump(final_summary, f, indent=2)
    print(f'Params saved for training mode "{training_mode}"')

    print(f"Training complete. Model saved to: {model_save_path}")

    # =========================================================================
    # DATA VIZ / RECAP
    # =========================================================================

    Compare(out_dir)