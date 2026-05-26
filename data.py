from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

import keras
import librosa

from signal_processing_tool import Signal


# --- Import Data ---------------------------------------------------------------------------------------

def import_data(X_path: str | Path, y_path: str | Path) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Import data from numpy files and return different informations about the data.
    
    Args:
        X_path (str | Path): Path to the data file containing samples. It is expected to be a .npy file.
        y_path (str | Path): Path to the data file containing labels. It is expected to be a .npy file.

    Returns:
        tuple[np.ndarray, np.ndarray, int, int, int]:\n 
        np.ndarray: Contains the data for each sample (1 row = 1 sample).\n
        np.ndarray: Contains the labels of each sample (1 row = 1 label).\n
        int: The number of samples of the dataset.\n
        int: The sample rate of the samples.\n
        int: The duration of each sample.\n
    """

    array_X = np.load(X_path)
    # This file 

    array_y = np.load(y_path)
    # This file contains the labels of each sample (1 row = 1 label)

    sr = 250 # Sampling rate is set to 250 Hz
    nb_samples = array_X.shape[0]
    duration = array_X.shape[1] / sr

    return array_X, array_y, nb_samples, sr, duration

# ---------------------------------------------------------------------------------------------------

# --- Model training data ---------------------------------------------------------------------------

def get_training_materials(array_X: np.ndarray, array_y: np.ndarray, nb_samples: int, sr: int) -> tuple:
    """Generates spectrograms, sorts them by class, and splits into training (2/3) and testing (1/3) sets.
    
    Args:
        array_X (np.ndarray): An array containing all samples from the dataset.
        array_y (np.ndarray): An array containing all labels from the dataset.
        nb_samples (int): The number of samples in the dataset.
        sr (int): The sampling rate of the samples.

    Returns:
        tuple[list, list, list, list]:\n
        np.ndarray: Samples used for training.\n
        np.ndarray: Coresponding labels.\n
        np.ndarray: Samples used for testing.\n
        np.ndarray: Coresponding labels.
    """

    X_train, y_train, X_test, y_test = [], [], [], []
    valid_classes = [0, 1, 2, 3, 4, 5, 6]  # Ignore class 7

    # Dictionary to store spectrograms and labels by class
    class_data = {i: [] for i in valid_classes}

    # Generate spectrograms only for valid classes
    print("Generating spectrograms ...")
    for s_id in range(nb_samples):
        label = array_y[s_id]
        if label in valid_classes:  # Only process valid classes
            current_sample = Signal(signal_name=f"S-{s_id}", signal_sr=sr, signal_X=array_X[s_id])
            current_sample.generate_spectrogram()

            # SpecAugment use
            S = current_sample.S[:,:, 0]
            if np.random.rand() < 0.5:
                S = spec_augment(S)

            class_data[label].append((current_sample.S, label))
        sys.stdout.write("Progress [%d%%]\r" % round(s_id/nb_samples * 100))
        sys.stdout.flush()

    # Split each class into training (2/3) and testing (1/3)
    for label in valid_classes:
        samples = class_data[label]
        np.random.shuffle(samples)  # Shuffle to ensure randomness
        split_idx = int(2/3 * len(samples))
        print(f"Label n°{label}: Nombre d'échantillons {len(samples)}")
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]

        # Add to training and testing lists
        for spectrogram, lbl in train_samples:
            X_train.append(spectrogram)
            y_train.append(lbl)
        for spectrogram, lbl in test_samples:
            X_test.append(spectrogram)
            y_test.append(lbl)

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    X_train = np.transpose(X_train, (0, 2, 3, 1))
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    X_test = np.transpose(X_test, (0, 2, 3, 1))
    y_test = np.array(y_test)

    # One-hot encode the labels
    y_train = keras.utils.to_categorical(y_train, num_classes=7)
    y_test = keras.utils.to_categorical(y_test, num_classes=7)

    # Verify shapes
    print(f"\nX_train shape: {X_train.shape} (type: {type(X_train)}), y_train shape: {y_train.shape} (type: {type(y_train)})")
    print(f"X_test shape: {X_test.shape} (type: {type(X_test)}), y_test shape: {y_test.shape} (type: {type(y_test)})")

    return X_train, y_train, X_test, y_test

def save_training_materials(X_train: list, y_train: list, X_test: list, y_test: list, save_dir: str | Path="saved_data") -> None:
    '''Saves the training and testing data to .npy files for later reuse.

    Args:
        X_train (np.ndarray): Training spectrograms.
        y_train (np.ndarray): Training labels (one-hot encoded).
        X_test (np.ndarray): Testing spectrograms.
        y_test (np.ndarray): Testing labels (one-hot encoded).
        save_dir (str): Directory where the files will be saved. Defaults to "saved_data".
    '''
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the data to .npy files
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)

    print(f"Data saved to {save_dir}/")

def load_training_materials(save_dir: str | Path="saved_data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Loads the training and testing data from previously saved .npy files.

    Args:
        save_dir (str): Directory where the files are stored. Defaults to "saved_data".

    Returns:
        tuple: (X_train, y_train, X_test, y_test) as numpy arrays.
    '''
    # Load the data from .npy files
    X_train = np.load(os.path.join(save_dir, "X_train.npy"))
    y_train = np.load(os.path.join(save_dir, "y_train.npy"))
    X_test = np.load(os.path.join(save_dir, "X_test.npy"))
    y_test = np.load(os.path.join(save_dir, "y_test.npy"))

    print(f"Data loaded from {save_dir}/")
    return X_train, y_train, X_test, y_test

# ---------------------------------------------------------------------------------------------------

# --- Training enhancement methods ------------------------------------------------------------------

def spec_augment(spec: np.ndarray, freq_mask: int=8, time_mask: int=8) -> np.ndarray:
    """A method to improve model's robustness. It applies masks to spectrograms during training data creation.
    Each call of the function will apply both a frequency and a time mask. freq_mask and time_mask correspond to
    the maximum length of the masks.
    Function returns the masked signal.
    
    Args:
        spec (np.ndarray): Spectrogram to be masked.
        freq_mask (int): maximum length of the frequency mask.
        time_mask (int): maximum length of the time mask.

    Returns:
        np.ndarray: The spctrogram with masks.
    """

    spec = spec.copy()

    freq_dim = spec.shape[0]
    time_dim = spec.shape[1]

    for _ in range(4):          # Apply several masks
        # Frequency mask :
        f_length = np.random.randint(0, min(freq_mask, freq_dim))
        if f_length > 0:
            f0 = np.random.randint(0, freq_dim - f_length + 1) # Get mask's position
            spec[:, f0 : f0 + f_length] = 0                      # Nullifies the part of the signal that correspond to the masks position

        # Time mask :
        t_length = np.random.randint(0, min(time_mask, time_dim))
        if t_length > 0:
            t0 = np.random.randint(0, time_dim - t_length + 1) # Get mask's position
            spec[:, t0 : t0 + t_length] = 0                      # Nullifies the part of the signal that correspond to the masks position
            

    return spec

# ---------------------------------------------------------------------------------------------------

# === Other files types handling ====================================================================
# --- mp3 files --------------------------------------

def mp3_to_npy(input_path: str | Path, output_path: str | Path, target_sr: int = 250, duration: float = 10.0) -> np.ndarray:
    """Convert a mp3 file to a npy file. The defaults settings are those of the WhhaleSounds dataset.

    Args:
        input_path (str | Path): Path to the mp3 file.
        output_path (str | Path): Path where the final npy file will be saved.
        target_sr (int): Target sampling rate (Hz). Default is the sr of the WhaleSounds dataset.
        duration (float): Duration of the audio segment to extract (seconds). Default is the duration of the samples of the WhaleSounds dataset.

    Returns:
        np.ndarray: The processed audio signal.
    """

    # Load the mp3 file
    signal, sr = librosa.load(input_path, sr=None)

    # Resample to the target sampling rate
    if sr != target_sr:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Trim the signal to the desired duration
    target_length = int(duration * target_sr)
    if len(signal) > target_length:
        # Trim the signal to the desired duration
        signal = signal[:target_length]
    elif len(signal) < target_length:
        # Pad the signal with zeros to reach the desired duration
        signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')

    # Normalize the signal
    signal = librosa.util.normalize(signal)

    # Save the signal as a npy file
    np.save(output_path, signal)

    return signal

def mp3_folder_to_npy(input_folder: str | Path, output_folder: str | Path, target_sr: int = 250, duration: float = 10.0) -> None:
    """Convert all mp3 files in a folder to npy files.

    Args:
        mp3_folder (str | Path): Path to the folder containing mp3 files.
        output_folder (str | Path): Path to the folder where npy files will be saved.
        target_sr (int): Target sampling rate (Hz). Default is the sr of the WhaleSounds dataset.
        duration (float): Duration of the audio segment to extract (seconds). Default is the duration of samples of the WhaleSounds dataset.
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all MP3 files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".npy"
            output_path = os.path.join(output_folder, output_filename)

            print(f"Converting {filename}...")
            mp3_to_npy(mp3_path, output_path, target_sr, duration)

    print("Conversion complete.")

# ----------------------------------------------------