import os
import numpy as np
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the path of the parent folder (BlueAria/) to the path, to be able to import data and signal


import keras

from data import *
"""data.py contains 4 functions :
import_data() : imports data from a dataset's X and y files, which paths are given. Returns both files as arrays, aswell
 as the number of samples, the sampling rate and the duration of the signals.
 
get_training_materials() : create every training elements (X and y _train, X and y _test) from imported data. Might take 
a while due to spectrogramification.
  
save_training materials() : save every training elements as .npy files in specified directory.

load_training_materials() : load every training elements saved in specified directory."""

from signal_processing_tool import signal


# DEBUG
def get_infos():
    samples_info = {
        'number of samples' : nb_samples,
        'number of classes' : len(np.unique(array_y)),
        'samples duration (sec)' : duration,
        'sampling rate (hz)' : sr
    }
    json_str = json.dumps(samples_info, indent=3)
    print(json_str)

    print("Dimensions de WhaleSounds_X :", array_X.shape)

    print("Dimensions de WhaleSounds_y :", array_y.shape)


# DEBUG
def display_test(id):
    s_test = signal(signal_name=f"Signal {id}", signal_sr=sr, signal_X=array_X[id])

    #s_test.filter(cutoff=50, order=4)
    s_test.generate_spectrogram()

    s_test.display()
    s_test.display(mode='spectrogram')





def generate_model(train = False, ep=50, patience=5):
    # Layers creation  :

    # Input layer :
    input_layer = keras.layers.Input(shape=(128, 40, 1))
    '''Settings explanation : 
        128 is the number of available frequencies (number of mels in spectrogram creation)
        2500 is time division (len(s_test.S[1]) = 20)
        1 correspond to the number of channels, here we only work with grey scale, so 1 channel is enough'''
    
    # First convolutional block
    conv1 = keras.layers.Conv2D(
        filters=32,
        kernel_size=(5, 3),  # Larger kernel in frequency axis
        padding='same',
        activation='relu'
    )(input_layer)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    #dropout1 = keras.layers.Dropout(0.25)(pool1)

    # Second convolutional block
    conv2 = keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    #dropout2 = keras.layers.Dropout(0.25)(pool2)

    # Third convolutional block
    conv3 = keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(pool2)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    #dropout3 = keras.layers.Dropout(0.25)(pool3)

    # Flatten and dense layers
    flatten = keras.layers.Flatten()(pool3)
    dense1 = keras.layers.Dense(units=128, activation='relu')(flatten)
    dropout4 = keras.layers.Dropout(0.5)(dense1)

    # Output layer
    output_layer = keras.layers.Dense(units=7, activation='softmax')(dropout4) # Units = number of classes


    model = keras.Model(input_layer, output_layer)
    model.summary()

    if train:
        if os.path.isdir("saved_data"):
            X_train, y_train, X_test, y_test = load_training_materials()
        else :
            X_train, y_train, X_test, y_test = get_training_materials(array_X, array_y, nb_samples, sr)
            save_training_materials(X_train, y_train, X_test, y_test)

        # EarlyStopping configuration : used to exit training early if a certain criteria is met (here on val_loss)
        callback = keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = patience,
            restore_best_weights=True
        )

        model.compile(optimizer='Adam',  # Good default optimizer to start with
                      loss='categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
                      metrics=['accuracy'])  # what to track
        
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=ep, callbacks=[callback])

        # Test the model:        
        val_loss, val_acc = model.evaluate(X_test, y_test)
        print(val_loss) # Error
        print(f"Accuracy: {val_acc} ({val_acc * 100}%)") # Accuracy

# === MAIN ===

array_X, array_y, nb_samples, sr, duration = import_data("D:/Cours/FabLab/fablab2026/datasets/WhaleSounds_huggingFace/WhaleSounds_X.npy", "D:/Cours/FabLab/fablab2026/datasets/WhaleSounds_huggingFace/WhaleSounds_y.npy")

generate_model(train=True, patience=6)