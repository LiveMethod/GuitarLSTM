import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


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
        ax1_bottom = divider.append_axes("bottom", size="55%", pad=0.25)

        # Custom divergent colormap from red to white to green
        rwg = [(0.7, 0, 0), (1, 1, 1), (0, 0.6, 0)]
        nodes = [0.0, 0.5, 1.0]
        diff_cmap = LinearSegmentedColormap.from_list('rwg', list(zip(nodes, rwg)))
        plt.colormaps.register(cmap=diff_cmap, name='rwg')

        # ==============================================================================
        # Top left (AX1): EQ Curve
        # ==============================================================================
        w_dry, h_dry = signal.freqz(dry)
        w_train, h_train = signal.freqz(train)
        w_pred, h_pred = signal.freqz(pred)

        ax1.set_title('Frequency Response (Log and linear)')
        ax1.plot(0.5 * sr * w_dry / np.pi, np.abs(h_dry), label='Dry', color='darkgray', linewidth=0.5, alpha=0.7)
        ax1.plot(0.5 * sr * w_train / np.pi, np.abs(h_train), label='Train', color='red', linewidth=0.5, alpha=0.6)
        ax1.plot(0.5 * sr * w_pred / np.pi, np.abs(h_pred), label='Pred', color='green', linewidth=0.5)
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

        ax1_bottom.plot(0.5 * sr * w_dry / np.pi, np.abs(h_dry), label='Dry', color='darkgray', linewidth=0.5, alpha=0.7)
        ax1_bottom.plot(0.5 * sr * w_train / np.pi, np.abs(h_train), label='Train', color='red', linewidth=0.5, alpha=0.6)
        ax1_bottom.plot(0.5 * sr * w_pred / np.pi, np.abs(h_pred), label='Pred', color='green', linewidth=0.5)
        ax1_bottom.set_ylim([min(np.abs(h_train))*0.5, max(np.abs(h_train))*1.5])
        ax1_bottom.set_xscale('linear')
        ax1_bottom.set_yscale('log')
        ax1_bottom.set_xlim([20, 20000])
        ax1_bottom.set_xticks([1, 1000, 5000, 10000, 15000, 20000])
        ax1_bottom.set_xticklabels(['0','1k', '5k', '10k', '15k', '20k'])

        # ==============================================================================
        # Bottom left (AX2): Power Spectral Density
        # ------------------------------------------------------------------------------
        # I'm gonna be real, I put this here by mistake while looking
        # for the EQ graph, and it felt more useful, so I kept it.
        # ==============================================================================

        f_dry, Pxx_dry = signal.welch(dry, sr, nperseg=1024)
        f_train, Pxx_train = signal.welch(train, sr, nperseg=1024)
        f_pred, Pxx_pred = signal.welch(pred, sr, nperseg=1024)
        
        ax2.set_title('Power Spectral Density')
        ax2.plot(f_dry, Pxx_dry, label='Dry', color='darkgray', linewidth=0.5, alpha=0.7)
        ax2.plot(f_train, Pxx_train, label='Train', color='red', linewidth=0.5, alpha=0.6)
        ax2.plot(f_pred, Pxx_pred, label='Pred', color='green', linewidth=0.5)
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
        ax3.set_title('dB Mel Spectrogram Diff')

        f, t, sgram_pred = signal.spectrogram(pred, sr, nperseg=1024, noverlap=512, mode='magnitude')
        f, t, sgram_train = signal.spectrogram(train, sr, nperseg=1024, noverlap=512, mode='magnitude')
        sgram_pred_db = 10 * np.log10(sgram_pred + 1e-10)
        sgram_train_db = 10 * np.log10(sgram_train + 1e-10)
        sgram_diff_db = sgram_pred_db - sgram_train_db

        # Plot the difference
        ax3.pcolormesh(t, f, sgram_diff_db, shading='auto', cmap='rwg', vmin=-10, vmax=10)
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

        # ==============================================================================
        # Bottom right (AX4): Waveform comparison
        # ==============================================================================
        overlap_waveform = np.minimum(pred, train)

        ax4.set_title('Waveform Comparison')
        ax4.plot(x_range, pred, label='Pred +', color='green', alpha=0.5)
        ax4.plot(x_range, train, label='Train +', color='red', alpha=0.5)
        ax4.plot(x_range, overlap_waveform, label='both', color='white', alpha=1)
        ax4.plot(x_range, dry, label='Dry', color='lightgray', alpha=1)
        ax4.legend()

        # Save the figure
        if(self.save):
            save_path = f'{self.out_dir}/result.png'
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        
        if(self.display):
            print("Displaying visualization...")
            plt.show()