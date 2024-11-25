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

    def dual_loss(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        """
        Combines base loss with frequency correction loss.
        """
        base_loss = Data.custom_loss(y_true, y_pred)
        freq_correction = Data.adaptive_freq_correction_loss(y_true, y_pred)
        return base_loss + 0.1 * freq_correction

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
      
      freqs = tf.linspace(20.0, 20000.0, tf.shape(stft_true)[-1])
      weights = tf.sqrt(freqs/1000.0)
      
      freq_diff = tf.abs(stft_true) - tf.abs(stft_pred)
      weighted_freq_loss = tf.reduce_mean(weights * tf.square(freq_diff))
      
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
    
    def mae(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
        """
        Mean absolute error (MAE)
        ---
        Measures the average magnitude of differences
        between predicted and actual values, without
        considering their direction (positive or negative).

        MAE uses abs() values rather than squares, making
        it less sensitive to outliers. This makes MAE
        particularly useful when you want to treat all
        deviations linearly rather than quadratically.

        Use MAE When:
        - You want to treat all errors proportionally
        - Outliers should not be overly penalized
        - Working with financial data or scenarios
          where the actual magnitude of error matters
        - Dealing with noisy data where occasional
          large errors are expected
        - The scale of your errors is already meaningful
          in your domain
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
        if isinstance(y, np.ndarray):
            return np.mean(np.abs(y - y_pred))
        else:
            return tf.reduce_mean(tf.abs(y - y_pred))
    #TODO: add input shape checking to ensure y and y_pred are compatible
    def erpen_loss(y:ArrayLike, y_pred:ArrayLike, use_filter=True, alpha:float = 0.5, beta:float = 0.5) -> float:
        """
        ERPen (Error Penalization) loss function
        ---
        Combines squared error and absolute error 
        terms with configurable weights.

        This hybrid loss function can be useful when you
        want to balance between MSE (Mean Squared Error)
        which penalizes larger errors more heavily and MAE
        (Mean Absolute Error) which is more robust to outliers

        Use when:
        - Fine-tuning the balance between MSE and MAE characteristics
        ---
        y: The true/target values (ground truth)
        y_pred: The predicted values from a model
        alpha: Weight for the squared error term (default 0.5)
        beta: Weight for the absolute error term (default 0.5)
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
        if isinstance(y, np.ndarray):
            return np.mean(alpha * np.square(y - y_pred) + beta * np.abs(y - y_pred))
        else:
            return tf.reduce_mean(alpha * tf.square(y - y_pred) + beta * tf.abs(y - y_pred))

    # def snr(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Signal to noise ratio (SNR)
    #     ---
    #     SNR measures the ratio of signal power to noise power.
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return 10 * np.log10(np.sum(np.square(y)) / np.sum(np.square(y - y_pred)))
    #     else:
    #         return 10 * tf.math.log(tf.reduce_sum(tf.square(y)) / tf.reduce_sum(tf.square(y - y_pred)))

    # def psnr(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Peak signal to noise ratio (PSNR)
    #     ---
    #     PSNR measures the ratio of peak signal power to noise power.
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return 10 * np.log10(np.max(np.square(y)) / np.mean(np.square(y - y_pred)))
    #     else:
    #         return 10 * tf.math.log(tf.reduce_max(tf.square(y)) / tf.reduce_mean(tf.square(y - y_pred)))
      
    def rms(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
        """
        Root mean squared error (RMSE)
        ---
        RMSE measures the square root of the average squared difference between predicted and actual signals.
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
        if isinstance(y, np.ndarray):
            return np.sqrt(np.mean(np.square(y - y_pred)))
        else:
            return tf.sqrt(tf.reduce_mean(tf.square(y - y_pred)))
    
    # def mape(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Mean absolute percentage error (MAPE)
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return np.mean(np.abs((y - y_pred) / y)) * 100
    #     else:
    #         return tf.reduce_mean(tf.abs((y - y_pred) / y)) * 100

    # def smape(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Symmetric mean absolute percentage error (SMAPE)
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return 100 * np.mean(2 * np.abs(y - y_pred) / (np.abs(y) + np.abs(y_pred)))
    #     else:
    #         return 100 * tf.reduce_mean(2 * tf.abs(y - y_pred) / (tf.abs(y) + tf.abs(y_pred)))
    
    # def mase(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Mean absolute scaled error (MASE)
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return np.mean(np.abs(y - y_pred)) / np.mean(np.abs(np.diff(y)))
    #     else:
    #         return tf.reduce_mean(tf.abs(y - y_pred)) / tf.reduce_mean(tf.abs(tf.diff(y)))

    # def tase(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Total absolute scaled error (TASE)
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return np.sum(np.abs(y - y_pred)) / np.sum(np.abs(np.diff(y)))
    #     else:
    #         return tf.reduce_sum(tf.abs(y - y_pred)) / tf.reduce_sum(tf.abs(tf.diff(y)))
    
    # def r2(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     R-squared (R2)
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(y)))
    #     else:
    #         return 1 - tf.reduce_sum(tf.square(y - y_pred)) / tf.reduce_sum(tf.square(y - tf.reduce_mean(y)))
    
    # def r2_adj(y:ArrayLike, y_pred:ArrayLike, n:int, p:int, use_filter=True) -> float:
    #     """
    #     Adjusted R-squared (R2_adj)
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return 1 - (n - 1) / (n - p - 1) * (1 - Data.r2(y, y_pred))
    #     else:
    #         return 1 - (n - 1) / (n - p - 1) * (1 - Data.r2(y, y_pred))

    # def mre(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Mean relative error (MRE)
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return np.mean(np.abs(y - y_pred) / y)
    #     else:
    #         return tf.reduce_mean(tf.abs(y - y_pred) / y)
    
    # def mrre(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Mean relative residual error (MRRE)
    #     --
    #     Use when: 
    #     - Error relative to signal magnitude matters more than absolute error
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return np.mean(np.abs(y - y_pred) / np.abs(y))
    #     else:
    #         return tf.reduce_mean(tf.abs(y - y_pred) / tf.abs(y))
    
    def huber_loss(y:ArrayLike, y_pred:ArrayLike, use_filter=True, delta:float = 1.0) -> float:
        """
        Huber loss
        ---
        Use Huber When:
        - you want MSE's sensitivity to small errors
          but MAE's robustness to outliers
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
        if isinstance(y, np.ndarray):
            return np.mean(np.where(np.abs(y - y_pred) < delta, 0.5 * np.square(y - y_pred), delta * (np.abs(y - y_pred) - 0.5 * delta)))
        else:
            return tf.reduce_mean(tf.where(tf.abs(y - y_pred) < delta, 0.5 * tf.square(y - y_pred), delta * (tf.abs(y - y_pred) - 0.5 * delta)))

    def smooth_l1_loss(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
        """
        Smooth L1 loss
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
        if isinstance(y, np.ndarray):
            return np.mean(np.where(np.abs(y - y_pred) < 1, 0.5 * np.square(y - y_pred), np.abs(y - y_pred) - 0.5))
        else:
            return tf.reduce_mean(tf.where(tf.abs(y - y_pred) < 1, 0.5 * tf.square(y - y_pred), tf.abs(y - y_pred) - 0.5))
    
    # def hinge_loss(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Hinge loss
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return np.mean(np.maximum(0, 1 - y * y_pred))
    #     else:
    #         return tf.reduce_mean(tf.maximum(0, 1 - y * y_pred))

    def log_cosh_loss(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
        """
        Log-cosh loss
        ---
        Similar benefits to Huber loss but with smoother gradients, good for audio waveform prediction
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
        if isinstance(y, np.ndarray):
            return np.mean(np.log(np.cosh(y_pred - y)))
        else:
            return tf.reduce_mean(tf.math.log(tf.math.cosh(y_pred - y)))
    
    
    def quantile_loss(y:ArrayLike, y_pred:ArrayLike, use_filter=True, q:float = 0.5) -> float:
        """
        Quantile loss
        --
        Use when:
        - Predicting intervals or dealing with asymmetric error costs
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
        if isinstance(y, np.ndarray):
            return np.mean(np.where(y >= y_pred, q * (y - y_pred), (1 - q) * (y_pred - y)))
        else:
            return tf.reduce_mean(tf.where(y >= y_pred, q * (y - y_pred), (1 - q) * (y_pred - y)))
    
    # def pinball_loss(y:ArrayLike, y_pred:ArrayLike, use_filter=True, tau:float = 0.5) -> float:
    #     """
    #     Pinball loss
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return np.mean(np.where(y >= y_pred, tau * (y - y_pred), (tau - 1) * (y_pred - y)))
    #     else:
    #         return tf.reduce_mean(tf.where(y >= y_pred, tau * (y - y_pred), (tau - 1) * (y_pred - y)))
    
    # def binary_crossentropy(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Binary crossentropy
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    #     else:
    #         return -tf.reduce_mean(y * tf.math.log(y_pred) + (1 - y) * tf.math.log(1 - y_pred))
    
    # def categorical_crossentropy(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Categorical crossentropy
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return -np.mean(np.sum(y * np.log(y_pred), axis=1))
    #     else:
    #         return -tf.reduce_mean(tf.reduce_sum(y * tf.math.log(y_pred), axis=1))
    
    # def sparse_categorical_crossentropy(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Sparse categorical crossentropy
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return -np.mean(np.log(np.choose(y, y_pred.T)))
    #     else:
    #         return -tf.reduce_mean(tf.math.log(tf.gather(y_pred, y, axis=1)))
    
    # def kullback_leibler_divergence(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
    #     """
    #     Kullback-Leibler divergence
    #     """
    #     if use_filter:
    #         y, y_pred = Data.pre_emphasis_highpass(y), Data.pre_emphasis_highpass(y_pred)
    #     if isinstance(y, np.ndarray):
    #         return np.mean(np.sum(y * np.log(y / y_pred), axis=1))
    #     else:
    #         return tf.reduce_mean(tf.reduce_sum(y * tf.math.log(y / y_pred), axis=1))
    
    

    def normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalize data to fit within the range of 
        either (0 to 1) or (-1 to 1) if neg values.
        """
        data_max = np.max(data)
        data_min = np.min(data)
        data_norm = max(data_max,abs(data_min))
        return data / data_norm