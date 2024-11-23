import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
import sys
from scipy import signal
import argparse
import struct
from mpl_toolkits.axes_grid1 import make_axes_locatable


def error_to_signal(y, y_pred, use_filter=1):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    if use_filter == 1:
        y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return np.sum(np.power(y - y_pred, 2)) / (np.sum(np.power(y, 2) + 1e-10))


def pre_emphasis_filter(x, coeff=0.95):
    return np.concatenate([x, np.subtract(x, np.multiply(x, coeff))])


def analyze_pred_vs_actual(args):
    try:
        output_wav = args.path + '/' + args.output_wav
        pred_wav = args.path + '/' + args.pred_wav
        input_wav = args.path + '/' + args.input_wav
        model_name = args.model_name
        show_plots = args.show_plots
        path = args.path
    except:
        output_wav = args['output_wav']
        pred_wav = args['pred_wav']
        input_wav = args['input_wav']
        model_name = args['model_name']
        show_plots = args['show_plots']
        path = args['path']

    # Read the input wav file
    sr_dry, y_dry = wavfile.read(os.path.join(os.getcwd(), input_wav))
    sr_train, y_train = wavfile.read(os.path.join(os.getcwd(), output_wav))
    sr_pred, y_pred = wavfile.read(os.path.join(os.getcwd(), pred_wav))

    # ==============================================================================
    # PREPROCESSING
    # ------------------------------------------------------------------------------
    # File lengths can sometimes differ by a sample or two. This will even
    # out a negligible difference but throw for anything noteworthy
    # ==============================================================================
    shortest_data = min(len(y_dry), len(y_train), len(y_pred))

    # Check if any signal length differs by more than 1%
    if abs(len(y_dry) - shortest_data) / shortest_data > 0.01 or \
       abs(len(y_train) - shortest_data) / shortest_data > 0.01 or \
       abs(len(y_pred) - shortest_data) / shortest_data > 0.01:
      raise ValueError("Signal lengths differ by more than 1%")

    # Shorten all signals to the shortest length
    # This is preferable to padding the short values because
    # it prevents zero pad vals from impacting analysis
    y_dry = y_dry[:shortest_data]
    y_train = y_train[:shortest_data]
    y_pred = y_pred[:shortest_data]

    # ==============================================================================
    # Viz setup
    # ------------------------------------------------------------------------------
    # Line graphs fix their Y ranges based on the training signal
    # so that graphs scale is identical across trainings, making
    # it easier to arrow through many files and compare results.
    # ==============================================================================
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(18, 10)

    x_range = np.linspace(0, len(y_dry) / sr_dry, num=len(y_dry))
    divider = make_axes_locatable(ax1)
    ax1_bottom = divider.append_axes("bottom", size="55%", pad=0.25)

    # ==============================================================================
    # Top left (AX1): EQ Curve
    # ==============================================================================
    w_dry, h_dry = signal.freqz(y_dry)
    w_train, h_train = signal.freqz(y_train)
    w_pred, h_pred = signal.freqz(y_pred)

    ax1.set_title('Frequency Response (Log and linear)')
    ax1.plot(0.5 * sr_train * w_dry / np.pi, np.abs(h_dry), label='Dry', color='darkgray', linewidth=0.5, alpha=0.7)
    ax1.plot(0.5 * sr_train * w_train / np.pi, np.abs(h_train), label='Train', color='red', linewidth=0.5, alpha=0.6)
    ax1.plot(0.5 * sr_pred * w_pred / np.pi, np.abs(h_pred), label='Pred', color='blue', linewidth=0.5)
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

    ax1_bottom.plot(0.5 * sr_train * w_dry / np.pi, np.abs(h_dry), label='Dry', color='darkgray', linewidth=0.5, alpha=0.7)
    ax1_bottom.plot(0.5 * sr_train * w_train / np.pi, np.abs(h_train), label='Train', color='red', linewidth=0.5, alpha=0.6)
    ax1_bottom.plot(0.5 * sr_pred * w_pred / np.pi, np.abs(h_pred), label='Pred', color='blue', linewidth=0.5)
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

    f_dry, Pxx_dry = signal.welch(y_dry, sr_dry, nperseg=1024)
    f_train, Pxx_train = signal.welch(y_train, sr_train, nperseg=1024)
    f_pred, Pxx_pred = signal.welch(y_pred, sr_pred, nperseg=1024)
    
    ax2.set_title('Power Spectral Density')
    ax2.plot(f_dry, Pxx_dry, label='Dry', color='darkgray', linewidth=0.5, alpha=0.7)
    ax2.plot(f_train, Pxx_train, label='Train', color='red', linewidth=0.5, alpha=0.6)
    ax2.plot(f_pred, Pxx_pred, label='Pred', color='blue', linewidth=0.5)
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

    f, t, sgram_pred = signal.spectrogram(y_pred, sr_pred, nperseg=1024, noverlap=512, mode='magnitude')
    f, t, sgram_train = signal.spectrogram(y_train, sr_train, nperseg=1024, noverlap=512, mode='magnitude')
    sgram_pred_db = 10 * np.log10(sgram_pred + 1e-10)
    sgram_train_db = 10 * np.log10(sgram_train + 1e-10)
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

    # ==============================================================================
    # Bottom right (AX4): Waveform comparison
    # ==============================================================================
    overlap_waveform = np.minimum(y_pred, y_train)

    ax4.set_title('Waveform Comparison')
    ax4.plot(x_range, y_pred, label='Pred +', color='blue', alpha=0.5)
    ax4.plot(x_range, y_train, label='Train +', color='red', alpha=0.5)
    ax4.plot(x_range, overlap_waveform, label='both', color='white', alpha=1)
    ax4.plot(x_range, y_dry, label='Dry', color='lightgray', alpha=1)
    ax4.legend()

    # ax4_bottom.plot(x_range, y_dry, label='Dry', color='darkgray', linewidth=0.5, alpha=0.7)
    # ax4_bottom.legend()
    # ax4_bottom.set_xticks([])
    # ax4_bottom.set_yticks([])
    
    # Save the figure
    # plt.savefig(os.path.join(path, model_name + '_frequency_domain.png'))


    if show_plots == 1:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--path", default=".")
    parser.add_argument("--output_wav", default=os.path.join("models/test/y_test.wav"))
    parser.add_argument("--pred_wav", default=os.path.join("models/test/y_pred.wav"))
    parser.add_argument("--input_wav", default=os.path.join("models/test/x_test.wav"))
    parser.add_argument("--model_name", default="plot")
    parser.add_argument("--path", default=os.getcwd())
    parser.add_argument("--show_plots", default=1)
    args = parser.parse_args()
    analyze_pred_vs_actual(args)