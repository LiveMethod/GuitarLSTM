import numpy as np
from scipy.signal import butter, lfilter
from scipy import interpolate

class Audio:
  def apply_high_cut(data, cutoff_freq, order=3):
    nyquist = 0.5 * 44100
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

  def apply_correction(data, eq_array):
    freq_data = np.fft.rfft(data)
    x = np.linspace(0, 1, len(eq_array))
    f = interpolate.CubicSpline(x, eq_array)
    x_new = np.linspace(0, 1, len(freq_data))
    scaled_eq_array = f(x_new)

    # Ensure eq_array is the same length as freq_data
    if len(scaled_eq_array) != len(freq_data):
      raise ValueError(f"scaled_eq_array must be the same length as the frequency data ({len(scaled_eq_array)}, {len(freq_data)})")
    
    corrected_freq_data = freq_data * scaled_eq_array
    corrected_data = np.fft.irfft(corrected_freq_data)

    return corrected_data