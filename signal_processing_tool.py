from __future__ import annotations

import numpy as np

import librosa
import librosa.display

import scipy.signal as sgnl

import matplotlib.pyplot as plt


class Signal:
    def __init__(self, signal_name: str, signal_X: list, signal_sr: int) -> Signal:
        """Create a signal object with several methods available.

        Args:
            signal_name (str): An identificator for the signal.
            signal_X (list): A list of the values constituting the signal.
            signal_sr (int): The sampling rate of the signal.

        Returns:
            Signal: The object.
        """

        s_array = np.array(signal_X) 
        s_array = librosa.util.normalize(s_array) # Normalize the signal, can help if the signal is weak
        self.X = s_array

        self.id = signal_name
        self.sr = signal_sr

        self.t = np.arange(len(self.X)) / self.sr
        self.sh_nyquist = self.sr / 2
    

    def filter(self, cutoff: float, order: int, type: str='low',) -> None:
        """Apply the designated filter to the signal. Type dictate which filter is applied : 
            'low' (low-pass)
            'high' (high-pass)

        Args:
            cutoff (float): The cutoff frequency for the filter.
            order (int): The order of the filter.
            type (str): Which filter to apply (low or high).
        """

        normalized_cutoff = cutoff / self.sh_nyquist
        b, a = sgnl.butter(order, normalized_cutoff, btype = type)
        # b and a contains the coefficients to apply on the signal.

        self.X = sgnl.filtfilt(b, a, self.X)


    def generate_spectrogram(self) -> None:
        """Generate the Mel spectrogram assiociated with the signal.
        """

        S = librosa.feature.melspectrogram(y=self.X, sr=self.sr, n_fft=512, hop_length=64, n_mels=128)
        # Settings can be adjusted to better handle low frequencies.

        # Get S but in dB instead of power.
        self.S = librosa.power_to_db(S, ref = np.max(S))
        self.S = (self.S - np.mean(self.S)) / (np.std(self.S) + 1e-6) # Normalize spect


    def display(self, mode: str='signal') -> None:
        """Display the signal using Matplotlib.

        Args:
            mode (str): What to display. Can either be the signal or its spectrogram.
        """

        if mode == 'signal':
            plt.figure(figsize=(14, 5))
            plt.plot(self.t, self.X, color='b')
            plt.title(f'Signal - {self.id}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.xlim(0, self.t[-1])

        elif mode == 'spectrogram':
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(self.S, sr = self.sr, x_axis = 'time', y_axis = 'hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Mel Spectrogram - {self.id}")
        
        plt.show()
