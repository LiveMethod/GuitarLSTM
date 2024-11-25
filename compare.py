import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from utils import Audio


class Compare:
    """
    Compare the output of the model to the source material.
    ---
    Generates data viz for freq response, power spectral density,
    mel spectrogram diff, and waveform comparison.

    Does not analyze metadata of the model file, which would be
    interesting as an additional output.
    """
    def __init__(self, out_dir,  save=True, display=False):
        self.save = save
        self.display = display
        if not save and not display:
            raise ValueError("Plot is set to neither save nor display")
        print((f"Generating visualization..."))
        self.validate(f'models/{out_dir}')
        self.analyze()

    def validate(self, out_dir):
        """
        Confirm that the path and its expected children exist.
        Valid results are stored on the class.
        """
        if not os.path.exists(out_dir):
            raise ValueError(f"Directory {out_dir} does not exist")
        else:
            self.out_dir = out_dir
        self.dry_wav = os.path.join(out_dir, 'x_test.wav')
        self.train_wav = os.path.join(out_dir, 'y_test.wav')
        self.pred_wav = os.path.join(out_dir, 'y_pred.wav')

        if not os.path.exists(self.dry_wav):
            raise ValueError(f"Input wav file {self.dry_wav} does not exist")
        if not os.path.exists(self.train_wav):
            raise ValueError(f"Output wav file {self.train_wav} does not exist")
        if not os.path.exists(self.pred_wav):
            raise ValueError(f"Prediction wav file {self.pred_wav} does not exist")
        
        # Load data, use a single known-good sample rate
        sr_d, dry = wavfile.read(self.dry_wav)
        sr_t, train = wavfile.read(self.train_wav)
        sr_p, pred = wavfile.read(self.pred_wav)
        if sr_d != sr_t or sr_t != sr_p:
            raise ValueError("Sample rates of the wav files do not match. Are they from the same session?")
        else:
            self.sr = sr_p
        
        # File lengths can sometimes differ by a sample or two. This will even
        # out a negligible difference but throw for anything noteworthy
        shortest_data = min(len(dry), len(train), len(pred))

        # Check if any signal length differs by more than 1%
        if abs(len(dry) - shortest_data) / shortest_data > 0.005 or \
        abs(len(train) - shortest_data) / shortest_data > 0.005 or \
        abs(len(pred) - shortest_data) / shortest_data > 0.005:
            raise ValueError("Signal lengths differ by more than 0.5%. Are they from the same session?")

        # Shorten all signals to the shortest length
        # This is preferable to padding the short values because
        # it prevents zero pad vals from impacting analysis
        self.dry = dry[:shortest_data]
        self.train = train[:shortest_data]
        self.pred = pred[:shortest_data]
        
    def analyze(self):
        dry, train, pred, sr = self.dry, self.train, self.pred, self.sr
        
        # ==============================================================================
        # Viz setup
        # ------------------------------------------------------------------------------
        # Line graphs fix their Y ranges based on the training signal
        # so that graphs scale is identical across trainings, making
        # it easier to arrow through many files and compare results.
        # ==============================================================================
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(18, 10)

        x_range = np.linspace(0, len(dry) / sr, num=len(dry))
        divider = make_axes_locatable(ax1)
        ax1_top = divider.append_axes("top", size="55%", pad=0.25)
        ax1_bottom = divider.append_axes("bottom", size="55%", pad=0.25)

        # Custom divergent colormap from red to white to green
        rwg = [(0.7, 0, 0), (1, 1, 1), (0, 0.6, 0)]
        nodes = [0.0, 0.5, 1.0]
        if 'rwg' not in plt.colormaps():
            diff_cmap = LinearSegmentedColormap.from_list('rwg', list(zip(nodes, rwg)))
            plt.colormaps.register(cmap=diff_cmap, name='rwg')

        # Basic stuff needed by both charts and re-q
        w_dry, h_dry = signal.freqz(dry)
        w_train, h_train = signal.freqz(train)
        w_pred, h_pred = signal.freqz(pred)
        
        pred_amplitudes = np.abs(h_pred)
        train_amplitudes = np.abs(h_train)
        # ==============================================================================
        # Find divergence point
        # ==============================================================================
        def find_divergence_points(curve1, curve2):
          """
          Find points where curves start diverging
          """
          window = 100
          correlations = np.array([
              np.corrcoef(curve1[i:i+window], curve2[i:i+window])[0,1]
              for i in range(len(curve1)-window)
          ])
          correlation_changes = np.gradient(correlations)
          corr_divergence = np.where(np.abs(correlation_changes) > np.std(correlation_changes))[0]
          
          return corr_divergence

        # Get frequency response curves between 1kHz and 20kHz
        hf_freq_mask = np.logical_and(0.5 * sr * w_pred / np.pi >= 1000, 0.5 * sr * w_pred / np.pi <= 20000)
        hf_pred_masked = np.abs(h_pred[hf_freq_mask])
        hf_pred_smooth = signal.savgol_filter(hf_pred_masked, window_length=100, polyorder=3)
        hf_train_masked = np.abs(h_train[hf_freq_mask])
        hf_train_smooth = signal.savgol_filter(hf_train_masked, window_length=100, polyorder=3)
        hf_divergences = find_divergence_points(hf_train_smooth, hf_pred_smooth)
        hf_eq_inflection = hf_divergences[0]
        hf_freqs = 0.5 * sr * w_pred[hf_freq_mask] / np.pi

        # ==============================================================================
        # Calculate EQ corrections
        # ==============================================================================
                # Calculate the difference in amplitude between y_pred and y_train in 500Hz buckets from 5kHz to 20kHz
        hf_bucket_size = 200
        hf_start_freq = hf_eq_inflection - 500
        hf_end_freq = 25000
        hf_num_buckets = (hf_end_freq - hf_start_freq) // hf_bucket_size
        hf_freq_buckets = np.linspace(hf_start_freq, hf_end_freq, hf_num_buckets + 1)

        lf_bucket_size = 24
        lf_start_freq = 0
        lf_end_freq = 400
        lf_num_buckets = (lf_end_freq - lf_start_freq) // lf_bucket_size
        lf_freq_buckets = np.linspace(lf_start_freq, lf_end_freq, lf_num_buckets + 1)

        hf_corrective_eq = np.ones_like(pred_amplitudes)
        lf_corrective_eq = np.ones_like(pred_amplitudes)
        
        for i in range(hf_num_buckets):
            bucket_start = hf_freq_buckets[i]
            bucket_end = hf_freq_buckets[i + 1]
            
            # Find indices corresponding to the current bucket
            bucket_indices = np.where((0.5 * sr * w_pred / np.pi >= bucket_start) & (0.5 * sr * w_pred / np.pi < bucket_end))
            
            # Calculate the average amplitude in the current bucket
            avg_pred_amp = np.mean(pred_amplitudes[bucket_indices])
            avg_train_amp = np.mean(train_amplitudes[bucket_indices])
            
            # Calculate the corrective factor
            if avg_pred_amp != 0:
              intensifier = 1
              if(bucket_start >= 12000):
                intensifier = np.interp(bucket_start, [12000, 20000], [1, 8])
              if avg_pred_amp < avg_train_amp:
                corrective_factor = (avg_train_amp / avg_pred_amp) * intensifier
              else:
                corrective_factor = (avg_train_amp / avg_pred_amp) / intensifier
            else:
              corrective_factor = 1
            
            # Apply the corrective factor to the EQ curve
            hf_corrective_eq[bucket_indices] *= corrective_factor
        
        for i in range(lf_num_buckets):
            bucket_start = lf_freq_buckets[i]
            bucket_end = lf_freq_buckets[i + 1]
            
            # Find indices corresponding to the current bucket
            bucket_indices = np.where((0.5 * sr * w_pred / np.pi >= bucket_start) & (0.5 * sr * w_pred / np.pi < bucket_end))
            
            # Calculate the average amplitude in the current bucket
            avg_pred_amp = np.mean(pred_amplitudes[bucket_indices])
            avg_train_amp = np.mean(train_amplitudes[bucket_indices])
            
            # Calculate the corrective factor
            if avg_pred_amp != 0:
              # intensifier = 0.7
              # No proven need for intensifier yet
              # if(bucket_start >= 12000):
              intensifier = np.interp(bucket_start, [-20, lf_end_freq], [6, 1.2])
              if avg_pred_amp < avg_train_amp:
                corrective_factor = (avg_train_amp / avg_pred_amp) * intensifier
              else:
                corrective_factor = (avg_train_amp / avg_pred_amp) / intensifier
            else:
              corrective_factor = 1
            
            # Apply the corrective factor to the EQ curve
            lf_corrective_eq[bucket_indices] *= corrective_factor

        # Smooth the corrective EQ curve
        hf_corrective_eq_smooth = signal.savgol_filter(hf_corrective_eq, 51, 3)
        lf_corrective_eq_smooth = signal.savgol_filter(lf_corrective_eq, 51, 3)
        # Merge the two smooth EQ curves
        # merged_corrective_eq_smooth = Audio.apply_high_cut(hf_corrective_eq_smooth * lf_corrective_eq_smooth, 16000)
        merged_corrective_eq_smooth = hf_corrective_eq_smooth * lf_corrective_eq_smooth
        # Apply the corrective EQ curve to the predicted signal
        corrected_wav = Audio.apply_high_cut(Audio.apply_correction(pred, merged_corrective_eq_smooth), 16000)
        
        # corrected_wav = Audio.apply_high_cut(pred, 5800)
        # Get the frequency response for the high cut filter

        # Create a signal with a perfectly flat frequency response
        signal_length = 1024  # Define the desired length
        flat_spectrum = np.ones(signal_length)
        flat_signal = np.fft.ifft(flat_spectrum).real
        flat_signal *= 10  # Boost by 10dB
        # w_high_cut, h_high_cut = signal.freqz(Audio.apply_high_cut(flat_signal, 5800))
        
        # corrected_wav = Audio.apply_correction(pred, hf_corrective_eq_smooth)
        w_corr, h_corr = signal.freqz(corrected_wav)
        w_eq, h_eq = signal.freqz(Audio.apply_correction(flat_signal, merged_corrective_eq_smooth))
        # Plot the frequency response of the high cut filter

        # Plot the demo curve
        ax1_top.plot(0.5 * sr * w_eq / np.pi, np.abs(h_eq), color='purple', linewidth=0.5, linestyle='--')
        ax1.plot(0.5 * sr * w_eq / np.pi, np.abs(h_eq), color='purple', linewidth=0.5, linestyle='--')

        # ==============================================================================
        # Top left (AX1): EQ Curve
        # ==============================================================================
        
        ax1_top.set_xlim([20, 20000])
        ax1_top.set_yscale('log')
        ax1_top.set_xscale('log')
        ax1_top.set_ylim([min(np.abs(h_train))*0.5, max(np.abs(h_train))*1.5])
        # Plot raw and smoothed graphs
        ax1_top.plot(hf_freqs, hf_pred_masked, color='limegreen', linewidth=0.5, solid_capstyle='round')
        ax1_top.plot(hf_freqs, hf_train_masked, color='pink', linewidth=0.5, solid_capstyle='round')
        ax1_top.plot(hf_freqs, hf_train_smooth, color='red', linewidth=0.5, alpha=0.7)
        ax1_top.plot(hf_freqs, hf_pred_smooth, color='green', linewidth=0.5, alpha=0.7)

        # Draw line at inflection point
        ax1_top.axvline(x=hf_freqs[hf_eq_inflection], color='red', alpha=1, linestyle='-', linewidth=1)
        
        # Plot the corrected EQ
        ax1.plot(0.5 * sr * w_corr / np.pi, np.abs(h_corr), label='Corrected', color='blue', linewidth=0.5, solid_capstyle='round')
        ax1_bottom.plot(0.5 * sr * w_corr / np.pi, np.abs(h_corr), label='Corrected', color='blue', linewidth=0.5, solid_capstyle='round')
        

        ax1.set_title('Frequency Response (Log and linear)')
        ax1.plot(0.5 * sr * w_dry / np.pi, np.abs(h_dry), label='Dry', color='darkgray', linewidth=0.5, alpha=0.7, solid_capstyle='round')
        ax1.plot(0.5 * sr * w_train / np.pi, np.abs(h_train), label='Train', color='red', linewidth=0.5, alpha=0.6, solid_capstyle='round')
        ax1.plot(0.5 * sr * w_pred / np.pi, np.abs(h_pred), label='Pred', color='green', linewidth=0.5, solid_capstyle='round')
        ax1.set_ylim([min(np.abs(h_train))*0.5, max(np.abs(h_train))*1.5])
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim([20, 20000])
        ax1.set_xticks([50, 100, 500, 1000, 5000, 10000, 20000])
        ax1.set_xticklabels(['50', '100', '500', '1k', '5k', '10k', '20k'])
        ax1.grid(which='both', axis='x', alpha=0.4)
        ax1.grid(which='major', axis='y', alpha=0.4)
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Amplitude')
        ax1.legend()

        ax1_bottom.plot(0.5 * sr * w_dry / np.pi, np.abs(h_dry), label='Dry', color='darkgray', linewidth=0.5, alpha=0.7, solid_capstyle='round')
        ax1_bottom.plot(0.5 * sr * w_train / np.pi, np.abs(h_train), label='Train', color='red', linewidth=0.5, alpha=0.6, solid_capstyle='round')
        ax1_bottom.plot(0.5 * sr * w_pred / np.pi, np.abs(h_pred), label='Pred', color='green', linewidth=0.5, solid_capstyle='round')
        ax1_bottom.set_ylim([min(np.abs(h_train))*0.5, max(np.abs(h_train))*1.5])
        ax1_bottom.set_xscale('linear')
        ax1_bottom.set_yscale('log')
        # ax1_bottom.set_xlim([0, 1000])
        ax1_bottom.set_xlim([20, 20000])
        ax1_bottom.set_xticks([1, 1000, 5000, 10000, 15000, 20000])
        ax1_bottom.set_xticklabels(['0','1k', '5k', '10k', '15k', '20k'])
        

        # ==============================================================================
        # Calculate inflection point where frequencies above 1k begin to diverge
        # ==============================================================================
        
        

        # ==============================================================================
        # Bottom left (AX2): Power Spectral Density
        # ------------------------------------------------------------------------------
        # I'm gonna be real, I put this here by mistake while looking
        # for the EQ graph, and it felt more useful, so I kept it.
        # ==============================================================================

        f_dry, Pxx_dry = signal.welch(dry, sr, nperseg=1024)
        f_train, Pxx_train = signal.welch(train, sr, nperseg=1024)
        f_pred, Pxx_pred = signal.welch(pred, sr, nperseg=1024)
        f_corr, Pxx_corr = signal.welch(corrected_wav, sr, nperseg=1024)
        
        ax2.set_title('Power Spectral Density')
        ax2.plot(f_dry, Pxx_dry, label='Dry', color='darkgray', linewidth=0.5, alpha=0.7)
        ax2.plot(f_train, Pxx_train, label='Train', color='red', linewidth=0.5, alpha=0.6)
        ax2.plot(f_pred, Pxx_pred, label='Pred', color='green', linewidth=0.5)
        # ax2.plot(f_corr, Pxx_corr, label='Corr', color='blue', linewidth=0.5)
        # Fix y axis based on train means identical graph scale across trainings
        ax2.set_ylim([min(Pxx_train)*0.5, max(Pxx_train)*1.5])
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim([20, 20000])
        ax2.set_xticks([50, 100, 500, 1000, 5000, 10000, 20000])
        ax2.set_xticklabels(['50', '100', '500', '1k', '5k', '10k', '20k'])
        ax2.grid(which='both', axis='x', alpha=0.4)
        ax2.grid(which='major', axis='y', alpha=0.4)
        ax2.set_ylabel('Power Density')        

        # ==============================================================================
        # Top right (AX3): dB Mel Spectrogram Diff
        # ==============================================================================
        # ax3.set_title('dB Mel Spectrogram Diff')

        f, t, sgram_pred = signal.spectrogram(pred, sr, nperseg=1024, noverlap=512, mode='magnitude')
        f, t, sgram_train = signal.spectrogram(train, sr, nperseg=1024, noverlap=512, mode='magnitude')
        f, t, sgram_corr = signal.spectrogram(corrected_wav, sr, nperseg=1024, noverlap=512, mode='magnitude')
        sgram_pred_db = 10 * np.log10(sgram_pred + 1e-10)
        sgram_train_db = 10 * np.log10(sgram_train + 1e-10)
        sgram_corr_db = 10 * np.log10(sgram_corr + 1e-10)
        # sgram_diff_db = sgram_pred_db - sgram_train_db
        sgram_diff_db = sgram_pred_db - sgram_train_db

        # Plot the difference
        ax3.pcolormesh(t, f, sgram_diff_db, shading='auto', cmap='bwr_r', vmin=-10, vmax=10)
        ax3.set_ylabel('Frequency [Hz]')
        ax3.set_xlabel('Time [sec]')
        ax3.set_yscale('log')
        ax3.set_ylim([20, 20000])
        ax3.set_yticks([50, 100, 500, 1000, 5000, 10000, 20000])
        ax3.set_yticklabels(['50', '100', '500', '1k', '5k', '10k', '20k'])
        ax3.grid(which='both', axis='y', alpha=0.4)
        # Add a legend for the color map
        cbar = plt.colorbar(ax3.collections[0], ax=ax3, orientation='vertical')
        cbar.set_label('Train+      Difference (dB)      Pred+')
        cbar.ax.set_yticklabels([])

        # # ==============================================================================
        # # Bottom right (AX4): Waveform comparison
        # # ==============================================================================
        shortest_len = min(len(pred), len(train))
        pred = pred[:shortest_len]
        train = train[:shortest_len]
        dry = dry[:shortest_len]
        x_range = np.linspace(0, len(pred) / sr, num=len(pred))
        overlap_waveform = np.minimum(pred, train)

        ax4.set_title('Waveform Comparison')
        ax4.plot(x_range, pred, label='Corr +', color='green', alpha=0.5)
        ax4.plot(x_range, train, label='Train +', color='red', alpha=0.5)
        ax4.plot(x_range, overlap_waveform, label='both', color='white', alpha=1)
        ax4.plot(x_range, dry, label='Dry', color='lightgray', alpha=1)
        ax4.legend()

        # Save the corrected wav file
        corrected_wav_path = os.path.join(self.out_dir, 'y_corr.wav')
        wavfile.write(corrected_wav_path, sr, corrected_wav)
        print(f"Saved corrected wav file to {corrected_wav_path}")

        # Save the figure
        if(self.save):
            save_path = f'{self.out_dir}/result.png'
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        
        if(self.display):
            print("Displaying visualization...")
            plt.show()