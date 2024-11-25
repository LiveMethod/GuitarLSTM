from ctypes import Array
import numpy as np
import tensorflow as tf
from typing import Union, cast

class Data:
    """
    Utilities for data shaping and manipulation
    ---
    These tend to accept numpy arrays or tensorflow tensors
    depending on where in the codebase they come from, so
    they should be writting to be compatible with both.
    """

    ArrayLike = Union[np.ndarray, tf.Tensor]
    def pre_emphasis_lowpass(x: ArrayLike, coeff: float = 1) -> ArrayLike:
        """
        Low-pass pre-emphasis filter
        ---
        Increases the relative amount of low-frequency signal,
        in effect boosting the prioritization of low-frequency
        information by the loss function, since it's slightly
        underrepresented in the input signal.
        """
        if isinstance(x, np.ndarray):
            return np.concatenate([x, np.add(x, np.multiply(x, coeff))])
        else:
            return tf.concat([x, x + coeff * x], 1)


    def pre_emphasis_highpass(x: ArrayLike, coeff: float = 0.95) -> ArrayLike:
        """
        High-boost pre-emphasis filter
        ---
        Increases the relative amount of high-frequency signal,
        in effect boosting the prioritization of high-frequency
        information by the loss function, since it's slightly
        overreoresented in the input signal.

        Useful because HF range is critical to guitar tone and
        one of the most difficult aspects to model convincingly.
        """
        if isinstance(x, np.ndarray):
            return np.concatenate([x, np.subtract(x, np.multiply(x, coeff))])
        else:
            return tf.concat([x, x - coeff * x], 1)

    def get_freq_weights(n_freqs: int, sample_rate: int = 44100) -> tf.Tensor:
      """
      Creates frequency weights with 10x emphasis above 5kHz
      """
      freqs = tf.linspace(0.0, sample_rate/2.0, n_freqs)
      weights = tf.ones_like(freqs)
      
      # Create mask for frequencies above 5kHz
      high_freq_mask = tf.cast(freqs > 5000.0, tf.float32)
      
      # Apply 10x multiplier above 5kHz
      weights = weights + (90.0 * high_freq_mask)  # 1.0 + 9.0 = 10x
      return weights

    def dual_loss(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        """
        Combines base loss with frequency correction loss.
        """
        base_loss = Data.custom_loss(y_true, y_pred)
        freq_correction = Data.adaptive_freq_correction_loss(y_true, y_pred)
        return base_loss + 0.1 * freq_correction

    def adaptive_freq_correction_loss(y_true: ArrayLike, y_pred: ArrayLike,
                                smoothing: float = 0.95,
                                min_freq: float = 20.0,
                                max_freq: float = 20000.0) -> ArrayLike:
        """
        Computes loss encouraging matching frequency response ratios.
        """
        was_numpy = isinstance(y_true, np.ndarray)
        
        # Convert to tensors and ensure 2D shape
        y_true_t = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred_t = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        if len(y_true_t.shape) == 1:
            y_true_t = tf.expand_dims(y_true_t, 0)
            y_pred_t = tf.expand_dims(y_pred_t, 0)

        # STFT processing
        stft_true = tf.signal.stft(y_true_t, frame_length=2048, frame_step=512)
        stft_pred = tf.signal.stft(y_pred_t, frame_length=2048, frame_step=512)

        mag_true = tf.abs(stft_true)
        mag_pred = tf.abs(stft_pred)

        # Calculate frequency ratio
        freq_ratio = (mag_true + 1e-6) / (mag_pred + 1e-6)

        # Apply enhanced high frequency weighting
        weights = Data.get_freq_weights(tf.shape(freq_ratio)[-1])
        freq_ratio = freq_ratio * tf.expand_dims(weights, 0)

        # Reshape for avg_pool1d: [batch, time, channels]
        freq_ratio = tf.transpose(freq_ratio, perm=[0, 2, 1])

        # Apply smoothing
        smoothed_ratio = tf.nn.avg_pool1d(
            freq_ratio,
            ksize=3,
            strides=1,
            padding='SAME'
        )

        ratio_loss = tf.reduce_mean(tf.abs(smoothed_ratio - 1.0))

        if was_numpy:
            return cast(Data.ArrayLike, ratio_loss.numpy())
        return cast(Data.ArrayLike, ratio_loss)

    def custom_loss(y_true: ArrayLike, y_pred: ArrayLike, 
                      sample_rate: int = 44100,
                      time_weight: float = 0.2, # time domain accuracy. + Transient shape, bounds 0.5 to 2.0
                      freq_weight: float = 3.0, # freq accuracy with emphasis on highs. + HF reproduction, +harshness -lowend, bounds 0.1 to 1.0
                      mel_weight: float = 0.2 # perceptual balance. + natural soundingness -harshness, bounds 0.1 to 0.5
                      ) -> ArrayLike:
      """
      Combined loss function handling both numpy arrays and tensorflow tensors.
      Returns same type as input.
      """
      was_numpy = isinstance(y_true, np.ndarray)

      # Convert to tensors for processing
      y_true_t = tf.convert_to_tensor(y_true, dtype=tf.float32)
      y_pred_t = tf.convert_to_tensor(y_pred, dtype=tf.float32)

      # Core loss calculation
      time_loss = tf.reduce_mean(tf.square(y_true_t - y_pred_t))

      stft_true = tf.signal.stft(y_true_t, frame_length=1024, frame_step=512)
      stft_pred = tf.signal.stft(y_pred_t, frame_length=1024, frame_step=512)

      mag_true = tf.abs(stft_true)
      mag_pred = tf.abs(stft_pred)

      freqs = tf.linspace(20.0, 20000.0, tf.shape(stft_true)[-1])
      # Get frequency weights
      weights = Data.get_freq_weights(tf.shape(mag_true)[-1], sample_rate)

      # Apply weighted frequency loss
      freq_diff = tf.abs(mag_true - mag_pred)
      weighted_freq_loss = freq_weight * tf.reduce_mean(weights * freq_diff)

      num_mel_bins = 128
      mel_matrix = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins=num_mel_bins,
          num_spectrogram_bins=tf.shape(stft_true)[-1],
          sample_rate=sample_rate,
          lower_edge_hertz=20.0,
          upper_edge_hertz=20000.0
      )

      mel_true = tf.tensordot(tf.abs(stft_true), mel_matrix, 1)
      mel_pred = tf.tensordot(tf.abs(stft_pred), mel_matrix, 1)
      mel_loss = tf.reduce_mean(tf.square(mel_true - mel_pred))

      total_loss = (time_weight * time_loss + 
                  freq_weight * weighted_freq_loss +
                  mel_weight * mel_loss)

      # Return same type as input
      if was_numpy:
          return cast(Data.ArrayLike, total_loss.numpy())
      return cast(Data.ArrayLike, total_loss)

    def esr(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
        """
        Error to signal ratio (ESR) 
        ---
        ESR measures the error between predicted and actual signals.
        https://www.mdpi.com/2076-3417/10/3/766/htm
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
        if isinstance(y, np.ndarray):
            return np.sum(np.power(y - y_pred, 2)) / (np.sum(np.power(y, 2) + 1e-10))
        else:
            return tf.reduce_sum(tf.square(y - y_pred)) / (tf.reduce_sum(tf.square(y)) + 1e-10)

    def mse(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
        """
        Mean squared error (MSE)
        ---
        Calculates the average of squared differences between
        predicted and actual values. This is a fundamental
        metric in machine learning, particularly useful for
        regression problems and model evaluation.

        Use MSE When:
        - Large errors are particularly undesirable
        - You want to heavily penalize outliers
        - Working with statistical models where squared
          errors are theoretically grounded
        - Training neural networks (MSE's smooth gradient
          helps optimization)
        - Dealing with scenarios where being way off is
          disproportionately worse than being a little off
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
        if isinstance(y, np.ndarray):
            return np.mean(np.square(y - y_pred))
        else:
            return tf.reduce_mean(tf.square(y - y_pred))

    def normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalize data to fit within the range of 
        either (0 to 1) or (-1 to 1) if neg values.
        """
        data_max = np.max(data)
        data_min = np.min(data)
        data_norm = max(data_max,abs(data_min))
        return data / data_norm