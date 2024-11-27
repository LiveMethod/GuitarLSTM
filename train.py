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
from keras._tf_keras.keras.losses import MeanSquaredError
import tf2onnx
import json

from compare import Compare
from utils import Data, IO

def main(
    training_mode:Literal['fast', 'accurate', 'extended'] = 'fast',
    split_data:int = 1,
    batch_size:int = 4096,
    input_size:int = 200,
    max_epochs:int = 3,
):

    timestamp = datetime.datetime.now().strftime("%H%M%S").lower()
    name="Asato 3x "
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

    for source in sources:
      for path in source:  # Skip the non-path element
        if not os.path.exists(path):
          raise FileNotFoundError(f"Error: Required audio file not found: {path}")

    conditioned_sources = []
    for source in sources:
      raw, conditioned = source
      cond_filename = conditioned.split('_')[-1].replace('.wav', '')
      parts = cond_filename.split('G')[1].split('T')
      g_val = parts[0]
      t_val, b_val = parts[1].split('B')
      conditioned_sources.append([[float(g_val), float(t_val), float(b_val)], raw, conditioned])

    control_names = ['gain', 'treble', 'bass']

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
        conv1d_strides = 3
        conv1d_filters = 32
        hidden_units = 64
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
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError(), metrics=[MeanSquaredError()])

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
    validation_c = [None for _ in control_names]
    validation_x = None
    validation_y = None
    

    for batch_num, (train_batch, test_batch) in enumerate(IO.load_conditioned_batch(conditioned_sources)):
        c_batch, x_batch, y_batch = train_batch
        c_val, x_val, y_val = test_batch
        print(f'Processing batch {batch_num + 1}...')


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
    
        #Store validation components
        if validation_x is None:
            if len(c_val) > 0 and len(x_val) > 0 and len(y_val) > 0:
              validation_c = c_val
              validation_x = x_val
              validation_y = y_val
            else:
              print("Validation data is empty and so is this batch")
        else:
            if len(c_val) > 0 and len(x_val) > 0 and len(y_val) > 0:
              validation_c = [np.concatenate([validation_c[i], c_val[i]]) for i in range(len(control_names))]
              # validation_c = np.concatenate([validation_c, c_val])
              validation_x = np.concatenate([validation_x, x_val])
              validation_y = np.concatenate([validation_y, y_val])
            else:
              print("Validation data is empty")
            
    
    # =========================================================================
    # PREDICT AND SAVE
    # =========================================================================
    model_save_path = f'models/{out_dir}/model.keras'
    final_summary = {
      **config_summary,
      "data_preprocessing": "normalized",
      "training_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
      "history": history.history,
    }

    # Ovbrwrite the config file with additional information
    with open(f'models/{out_dir}/summary.json', 'w') as f:
      json.dump(final_summary, f, indent=2)
    print(f'Params saved for training mode "{training_mode}"')

    # sys.stderr = open(os.devnull, "w")  # silence stderr
    model.save(model_save_path)
    # sys.stderr = sys.__stderr__  # unsilence stderr
    print(f"Training complete. Model saved to: {model_save_path}")

    # Define the TensorSpec for each input
    spec = (
        tf.TensorSpec(model.inputs[0].shape, tf.float32, name="audio_input"),
        tf.TensorSpec(model.inputs[1].shape, tf.float32, name="gain"),
        tf.TensorSpec(model.inputs[2].shape, tf.float32, name="treble"),
        tf.TensorSpec(model.inputs[3].shape, tf.float32, name="bass"),
    )

    # Convert Keras model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    # Save the ONNX model to a file
    with open(f"models/{out_dir}/model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("Model saved in ONNX format to 'model.onnx'")

    # Run Prediction #################################################
    print("Running prediction...")

    x_wav = validation_x[input_size-1:]
    y_wav = validation_y[input_size-1:]
    
    # In a past version this was input_size +1 at the end of the second arange. Unclear if that's needed
    indices = np.arange(input_size) + np.arange(len(validation_x)-input_size)[:,np.newaxis]
    x_test = tf.gather(validation_x,indices)
    
    control_values = {name: tf.gather(validation_c[i], indices) for i, name in enumerate(control_names)}
    
    # c_2 = Data.create_conditioning_array(0.2, indices, input_size)
    c_2 = np.full((len(validation_x), 1), [0.2])
    c2 = tf.gather(c_2, indices)
    # c_5 = Data.create_conditioning_array(0.5, indices, input_size)
    c_5 = np.full((len(validation_x), 1), [0.5])
    c5 = tf.gather(c_5, indices)
    # c_9 = Data.create_conditioning_array(0.9, indices, input_size)
    c_9 = np.full((len(validation_x), 1), [0.9])
    c9 = tf.gather(c_9, indices)

    ramp_up = np.linspace(0, 1, len(validation_x) // 2)
    ramp_down = np.linspace(1, 0, len(validation_x) - len(ramp_up))
    c_misc = np.concatenate([ramp_up, ramp_down])
    c_misc = c_misc[:, np.newaxis]  # Reshape to match the expected input shape
    c_misc = tf.gather(c_misc, indices)
    
    pred_gain = model.predict(x={'audio_input': x_test, 'gain': c_misc, 'treble': c5, 'bass': c5}, batch_size=batch_size)
    wavfile.write(f'models/{out_dir}/y_ramp_gain.wav', sr, pred_gain.flatten().astype(np.float32))
    
    pred_2 = model.predict(x={'audio_input': x_test, 'gain': c2, 'treble': c2, 'bass': c2}, batch_size=batch_size)
    wavfile.write(f'models/{out_dir}/y_all_2.wav', sr, pred_2.flatten().astype(np.float32))

    pred_5 = model.predict(x={'audio_input': x_test, 'gain': c5, 'treble': c5, 'bass': c5}, batch_size=batch_size)
    wavfile.write(f'models/{out_dir}/y_all_5.wav', sr, pred_5.flatten().astype(np.float32))

    pred_9 = model.predict(x={'audio_input': x_test, 'gain': c9, 'treble': c9, 'bass': c9}, batch_size=batch_size)
    wavfile.write(f'models/{out_dir}/y_all_9.wav', sr, pred_9.flatten().astype(np.float32))

    pred_treble = model.predict(x={'audio_input': x_test, 'gain': c2, 'treble': c_misc, 'bass': c5}, batch_size=batch_size)
    wavfile.write(f'models/{out_dir}/y_ramp_treb.wav', sr, pred_treble.flatten().astype(np.float32))

    pred_bass = model.predict(x={'audio_input': x_test, 'gain': c2, 'treble': c5, 'bass': c_misc}, batch_size=batch_size)
    wavfile.write(f'models/{out_dir}/y_ramp_bass.wav', sr, pred_bass.flatten().astype(np.float32))


    # pred2 = model.predict(x={'audio_input': x_test, 'control_input': c2}, batch_size=batch_size)
    # wavfile.write(f'models/{out_dir}/y_pred2.wav', sr, pred2.flatten().astype(np.float32))

    # pred5 = model.predict({'audio_input': x_test, 'control_input': c5}, batch_size=batch_size)
    # wavfile.write(f'models/{out_dir}/y_pred5.wav', sr, pred5.flatten().astype(np.float32))

    # pred9 = model.predict({'audio_input': x_test, 'control_input': c9}, batch_size=batch_size)
    # wavfile.write(f'models/{out_dir}/y_pred9.wav', sr, pred9.flatten().astype(np.float32))

    wavfile.write(f'models/{out_dir}/x_test.wav', sr, x_wav.flatten().astype(np.float32))
    wavfile.write(f'models/{out_dir}/y_test.wav', sr, y_wav.flatten().astype(np.float32))
    print("Prediction complete \n")

    

    # =========================================================================
    # DATA VIZ / RECAP
    # =========================================================================

    # Compare(out_dir)