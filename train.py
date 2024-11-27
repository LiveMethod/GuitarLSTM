import os
import datetime
import time
import sys
from typing import Literal
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from keras import Sequential
from keras._tf_keras.keras.layers import LSTM, Conv1D, Dense, Input, Concatenate, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.backend import clear_session
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import TimeDistributed
import json

from compare import Compare
from utils import Data, IO

def main(
    training_mode:Literal['fast', 'accurate', 'extended'] = 'fast',
    split_data:int = 1,
    batch_size:int = 4096*2,
    input_size:int = 100,
    max_epochs:int = 1,
):

    timestamp = datetime.datetime.now().strftime("%H%M%S").lower()
    name="multi silver all "
    out_dir = f'{timestamp}'
    os.makedirs(f'models/{out_dir}')
    print(f"Current output directory: {out_dir}")

    sources = [
      ["data/asato_conditioned/A_raw.wav", "data/asato_conditioned/A_G0T0B0.wav"],
      ["data/asato_conditioned/B_raw.wav", "data/asato_conditioned/B_G26T50B35.wav"],
      ["data/asato_conditioned/C_raw.wav", "data/asato_conditioned/C_G45T26B100.wav"],
      ["data/asato_conditioned/D_raw.wav", "data/asato_conditioned/D_G79T82B11.wav"],
      ["data/asato_conditioned/E_raw.wav", "data/asato_conditioned/E_G15T53B67.wav"],
      ["data/asato_conditioned/F_raw.wav", "data/asato_conditioned/F_G100T74B52.wav"],
      ["data/asato_conditioned/G_raw.wav", "data/asato_conditioned/G_G65T14B26.wav"],
      ["data/asato_conditioned/H_raw.wav", "data/asato_conditioned/H_G39T100B82.wav"],
      ["data/asato_conditioned/I_raw.wav", "data/asato_conditioned/I_G76T47B33.wav"],
      ["data/asato_conditioned/J_raw.wav", "data/asato_conditioned/J_G26T36B55.wav"]
    ]
    # sources = [
      # [[0.0], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_00.wav"],
      # [[0.1], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_01.wav"],
      # [[0.2], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_02.wav"],
      # [[0.3], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_03.wav"],
      # [[0.4], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_04.wav"],
      # [[0.5], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_04.wav"],
      # [[0.6], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_06.wav"],
      # [[0.7], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_07.wav"],
      # [[0.8], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_08.wav"],
      # [[0.9], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_09.wav"],
      # [[1.0], "data/silver_conditioned/raw.wav", "data/silver_conditioned/c_gain_10.wav"]
    # ]
    for source in sources:
      for path in source:  # Skip the non-path element
        if not os.path.exists(path):
          raise FileNotFoundError(f"Error: Required audio file not found: {path}")

    conditioned_sources = []
    for source in sources:
      raw, conditioned = source
      # print(source)
      # print(raw, conditioned)
      cond_filename = conditioned.split('_')[-1].replace('.wav', '')
      parts = cond_filename.split('G')[1].split('T')
      g_val = parts[0]
      t_val, b_val = parts[1].split('B')
      conditioned_sources.append([[float(g_val), float(t_val), float(b_val)], raw, conditioned])
    cond_count = len(conditioned_sources[0][0])
    control_names = ['gain', 'tone', 'bass']

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

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
        conv1d_filters = 16
        hidden_units = 50
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
    audio_input = Input(shape=(input_size, 1), name='audio_input')
    control_inputs = {name: Input(shape=(input_size, 1), name=name) for name in control_names}
    merged_input = Concatenate(axis=-1)([audio_input] + list(control_inputs.values()))
    x = Conv1D(conv1d_filters, kernel_size=kernel_size, strides=conv1d_strides, activation='relu', padding='same')(merged_input)
    x = Conv1D(conv1d_filters, kernel_size=kernel_size, strides=conv1d_strides, activation='relu', padding='same')(x)
    # x = Dropout(0.2)(x)
    x = LSTM(hidden_units)(x)
    # x = Dropout(0.2)(x)
    output = Dense(1, activation=None)(x)
    control_layers = {control_name: control_inputs[control_name] for control_name in control_names}
    model = Model(inputs={'audio_input': audio_input, **control_layers}, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Data.dual_loss, metrics=[Data.dual_loss])

    # print('\n') ;model.summary(); print('\n')

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
    validation_c = None
    validation_x = None
    validation_y = None
    

    for batch_num, (train_batch, test_batch) in enumerate(IO.load_conditioned_batch(conditioned_sources)):
        c_batch, x_batch, y_batch = train_batch
        c_val, x_val, y_val = test_batch
        print(f'Processing batch {batch_num + 1}...')

        # print("166 x", x_batch.shape)
        # print("167 c", c_batch[0].shape)

        # Prepare data
        y_ordered = y_batch[input_size - 1:]
        indices = np.arange(input_size) + np.arange(len(x_batch) - input_size + 1)[:, np.newaxis]
        x_ordered = tf.gather(x_batch, indices)
        shuffled_indices = np.random.permutation(len(x_ordered))

        # Shuffle data
        x_random = tf.gather(x_ordered, shuffled_indices)
        y_random = tf.gather(y_ordered, shuffled_indices)
        all_c_random = []
        for ci in c_batch:  
          ci_ordered = tf.gather(ci, indices)
          # ci_random = tf.expand_dims(tf.gather(ci_ordered, shuffled_indices), axis=-1)
          ci_random = tf.gather(ci_ordered, shuffled_indices)
          all_c_random.append(ci_random)
        
        c_random = {name: all_c_random[i] for i, name in enumerate(control_names)}
    
        # Train on batch
        training_data = {'audio_input': x_random, **c_random}
        print("Training data shapes:")
        print("(11995101, 100, 1) is a valid example")
        for key, value in training_data.items():
          print(f'{key}: {value.shape}')
        
        batch_history = model.fit(
            training_data,
            y=y_random,
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
    
        # Store validation components
        # if validation_x is None:
        #     if len(c_val) > 0 and len(x_val) > 0 and len(y_val) > 0:
        #       validation_c = c_val
        #       validation_x = x_val
        #       validation_y = y_val
        #     else:
        #       print("Validation data is empty and so is this batch")
        # else:
        #     if len(c_val) > 0 and len(x_val) > 0 and len(y_val) > 0:
        #       validation_c = np.concatenate([validation_c, c_val])
        #       validation_x = np.concatenate([validation_x, x_val])
        #       validation_y = np.concatenate([validation_y, y_val])
        #     else:
        #       print("Validation data is empty")
            
    
    # =========================================================================
    # PREDICT AND SAVE
    # =========================================================================
    model_save_path = f'models/{out_dir}/model.h5'
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

    sys.stderr = open(os.devnull, "w")  # silence stderr
    model.save(model_save_path)
    sys.stderr = sys.__stderr__  # unsilence stderr
    print(f"Training complete. Model saved to: {model_save_path}")

    # Run Prediction #################################################
    print("Running prediction...")

    # x_wav = validation_x[input_size-1:]
    # y_wav = validation_y[input_size-1:]
    
    # # In a past version this was input_size +1 at the end of the second arange. Unclear if that's needed
    # indices = np.arange(input_size) + np.arange(len(validation_x)-input_size)[:,np.newaxis]
    # x_test = tf.gather(validation_x,indices)
    # c_test = tf.gather(validation_c,indices)
    
    # # c_2 = Data.create_conditioning_array(0.2, indices, input_size)
    # c_2 = np.full((len(validation_c), 1), [0.2])
    # c2 = tf.gather(c_2, indices)
    # # c_5 = Data.create_conditioning_array(0.5, indices, input_size)
    # c_5 = np.full((len(validation_c), 1), [0.5])
    # c5 = tf.gather(c_5, indices)
    # # c_9 = Data.create_conditioning_array(0.9, indices, input_size)
    # c_9 = np.full((len(validation_c), 1), [0.9])
    # c9 = tf.gather(c_9, indices)

    # ramp_up = np.linspace(0, 1, len(validation_c) // 2)
    # ramp_down = np.linspace(1, 0, len(validation_c) - len(ramp_up))
    # c_misc = np.concatenate([ramp_up, ramp_down])
    # c_misc = c_misc[:, np.newaxis]  # Reshape to match the expected input shape
    # c_misc = tf.gather(c_misc, indices)

    # pred_all = model.predict(x={'audio_input': x_test, 'control_input': c_misc}, batch_size=batch_size)
    # wavfile.write(f'models/{out_dir}/y_pred_all.wav', sr, pred_all.flatten().astype(np.float32))

    # pred2 = model.predict(x={'audio_input': x_test, 'control_input': c2}, batch_size=batch_size)
    # wavfile.write(f'models/{out_dir}/y_pred2.wav', sr, pred2.flatten().astype(np.float32))

    # pred5 = model.predict({'audio_input': x_test, 'control_input': c5}, batch_size=batch_size)
    # wavfile.write(f'models/{out_dir}/y_pred5.wav', sr, pred5.flatten().astype(np.float32))

    # pred9 = model.predict({'audio_input': x_test, 'control_input': c9}, batch_size=batch_size)
    # wavfile.write(f'models/{out_dir}/y_pred9.wav', sr, pred9.flatten().astype(np.float32))

    # wavfile.write(f'models/{out_dir}/x_test.wav', sr, x_wav.flatten().astype(np.float32))
    # wavfile.write(f'models/{out_dir}/y_test.wav', sr, y_wav.flatten().astype(np.float32))
    # print("Prediction complete \n")

    

    # =========================================================================
    # DATA VIZ / RECAP
    # =========================================================================

    # Compare(out_dir)