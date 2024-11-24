from ctypes import Array
import numpy as np
import tensorflow as tf
from typing import Union

class Data:
    """
    Utilities for data shaping and manipulation
    ---
    These tend to accept numpy arrays or tensorflow tensors
    depending on where in the codebase they come from, so
    they should be writting to be compatible with both.
    """

    ArrayLike = Union[np.ndarray, tf.Tensor]

    def pre_emphasis_filter(x: ArrayLike, coeff: float = 0.95) -> ArrayLike:
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

    def esr(y:ArrayLike, y_pred:ArrayLike, use_filter=True) -> float:
        """
        Error to signal ratio (ESR) 
        ---
        ESR measures the error between predicted and actual signals.
        https://www.mdpi.com/2076-3417/10/3/766/htm
        """
        if use_filter:
            y, y_pred = Data.pre_emphasis_filter(y), Data.pre_emphasis_filter(y_pred)
        if isinstance(y, np.ndarray):
            return np.sum(np.power(y - y_pred, 2)) / (np.sum(np.power(y, 2) + 1e-10))
        else:
            return tf.reduce_sum(tf.square(y - y_pred)) / (tf.reduce_sum(tf.square(y)) + 1e-10)
    
    def normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalize data to fit within the range of 
        either (0 to 1) or (-1 to 1) if neg values.
        """
        data_max = np.max(data)
        data_min = np.min(data)
        data_norm = max(data_max,abs(data_min))
        return data / data_norm