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





def generate_model(train = False, ep=50, patience=5, data_file="saved_data"):
    # Layers creation  :

    # Input layer :
    input_layer = keras.layers.Input(shape=(64, 20, 1))
    '''Settings explanation : 
        128 is the number of available frequencies (number of mels in spectrogram creation)
        2500 is time division (len(s_test.S[1]) = 20)
        1 correspond to the number of channels, here we only work with grey scale, so 1 channel is enough'''
    
    # First convolutional block, specialised in finding frequency motives
    conv1 = keras.layers.Conv2D(
        filters=32,
        kernel_size=(5, 3),  # Larger kernel in frequency axis
        padding='same'
    )(input_layer)
    bn1 = keras.layers.BatchNormalization()(conv1)
    act1 = keras.layers.Activation('relu')(bn1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 1))(act1)

    # Second convolutional block
    conv2 = keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same'
    )(pool1)
    bn2 = keras.layers.BatchNormalization()(conv2)
    act2 = keras.layers.Activation('relu')(bn2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 1))(act2)
    spDropout1 = keras.layers.SpatialDropout2D(0.2)(pool2) # Classic dropout could break motives, we instead use this

    # Third convolutional block, specialised in finding time motives
    conv3 = keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 5), # Larger kernel in time axis
        padding='same'
    )(spDropout1)
    bn3 = keras.layers.BatchNormalization()(conv3)
    act3 = keras.layers.Activation('relu')(bn3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(act3)
    spDropout2 = keras.layers.SpatialDropout2D(0.3)(pool3)

    # Fourth convolutional block
    conv4 = keras.layers.Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding='same'
    )(spDropout2)
    bn4 = keras.layers.BatchNormalization()(conv4)
    act4= keras.layers.Activation('relu')(bn4)
    pool4 = keras.layers.MaxPooling2D((2,2))(act4)

    # Global Average Pooling layer
    gAvgPool = keras.layers.GlobalAveragePooling2D()(pool4)

    # Dense layer & dropout
    dense1 = keras.layers.Dense(units=128, activation='relu')(gAvgPool)
    dropout4 = keras.layers.Dropout(0.5)(dense1)

    # Output layer
    output_layer = keras.layers.Dense(units=7, activation='softmax')(dropout4) # Units = number of classes


    model = keras.Model(input_layer, output_layer)
    model.summary()

    if train:
        if os.path.isdir(data_file):
            X_train, y_train, X_test, y_test = load_training_materials()
        else :
            X_train, y_train, X_test, y_test = get_training_materials(array_X, array_y, nb_samples, sr)
            save_training_materials(X_train, y_train, X_test, y_test, save_dir=data_file)

        # CALLBACKS
        # EarlyStopping configuration : used to exit training early if a certain criteria is met (here on val_loss)
        earlyStop = keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = patience,
            restore_best_weights=True
        )

        # ReduceLROnPlateau
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = 0.5,
            patience = 3
        )

        optimizer = keras.optimizers.Adam(  # Good default optimizer to start with
            learning_rate=0.0005    # Default is 0.001, this value is often better. The learning rate will be adjusted during training thanks to the reduceLRonPlateau method.
        )

        model.compile(optimizer=optimizer,  
                      loss='categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
                      metrics=['accuracy'])  # what to track
        
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=ep, callbacks=[earlyStop, lr_scheduler])

        # Test the model:        
        val_loss, val_acc = model.evaluate(X_test, y_test)
        print(val_loss) # Error
        print(f"Accuracy: {val_acc} ({val_acc * 100}%)") # Accuracy

# === MAIN ===

array_X, array_y, nb_samples, sr, duration = import_data("D:/Cours/FabLab/fablab2026/datasets/WhaleSounds_huggingFace/WhaleSounds_X.npy", "D:/Cours/FabLab/fablab2026/datasets/WhaleSounds_huggingFace/WhaleSounds_y.npy")

generate_model(train=True, ep=50, patience=6, data_file="data_2ndSpecSettings")