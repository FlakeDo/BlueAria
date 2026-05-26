from __future__ import annotations

from pathlib import Path
import keras
from data import load_training_materials
import numpy as np
import wave

# Models (both Keras models and YOLO models) can be tested here. Models are saved in 'model' diretories within Keras and YOLO folders and 
# are saved as <(library)_cnn_v(model.version)>.

LAST_KERAS_VERSION = "K10"
LAST_YOLO_VERSION = "Y01"

def import_model(model_version: str=LAST_KERAS_VERSION) -> keras.src.models.functional.Functional:
    """Returns a model which version (as <(FIRST_LETTER_OF_LIBRARY)(version_without_.)>) was given, regardless of if it's a YOLO or a Keras model.

    Args:
        model_version (str): A model version, expected as "<Y or K><version number without . (ex. 1.0 -> 10)>"

    Returns:
        keras.src.models.functional.Functional: Keras model
    """

    #  YOLO IMPORT NOT YET IMPLEMENTED.

    model = keras.models.load_model("bluearia_Keras/model/bluearia_" + model_version + ".keras")
    print(f"Model {model_version} imported successfully.")

    return model


def test_model(model: keras.src.models.functional.Functional, data_dir: str | Path="data_default") -> None:
    """Test a given model, regardless of its type. It will test the model on test samples from the Hugging Face dataset. For testing on something else, please 
    see test_model_on_foreign().

    Args:
        Model (keras.src.models.functional.Functional): The model which will be tested.
        data_dir (str | Path): Where testing data will be fetched.
    """

    _, _, X_test, y_test =load_training_materials(data_dir)

    predictions = model.predict(X_test)

    # Display prediction results :

    good_predict = 0

    for i in range(len(predictions)):
        predicted_class = np.argmax(predictions[i])
        true_class = np.argmax(y_test[i])

        if predicted_class == true_class:
            good_predict += 1
    
    print("Overall predictions accuracy : ", good_predict/len(predictions) * 100, "%")


def test_model_on_foreign(model: keras.src.models.functional.Functional, data_dir: str | Path) -> None:
    """Test a given model, regardless of its type. It will test the model on test samples from the given dataset.

    Args:
        Model (keras.src.models.functional.Functional): The model which will be tested.
        data_dir (str | Path): Where testing data will be fetched.
    """

    _, _, X_test, y_test =load_training_materials(data_dir)

    predictions = model.predict(X_test)

    # Display prediction results :

    good_predict = 0

    for i in range(len(predictions)):
        predicted_class = np.argmax(predictions[i])
        true_class = np.argmax(y_test[i])

        if predicted_class == true_class:
            good_predict += 1
    
    print("Overall predictions accuracy : ", good_predict/len(predictions) * 100, "%")


# --- Model use ---------------------------------------------------------

def wav_to_array(wav_path: str | Path) -> np.ndarray:
    """Convert a wav file into a numpy array.

    Args:
        wav_path (str | Path): The path where the wav file is located.

    Returns:
        np.ndarray: Given file turned into an array.
    """

    # Read file
    file = wave.open(wav_path)
    samples = file.getnframes()
    audio = file.readframes(samples)

    # Convert to float32
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise (between -1 & 1)
    audio_normalised = audio_as_np_float32 / np.iinfo(np.int16).max

    return audio_normalised


def use_model(model: keras.src.models.functional.Functional, arg2: str) -> None:
    """Use the specified model to predict a sample's class.
    
    Args:
        Model (keras.src.models.functional.Functional): The model which will be used.
        arg2 (str): The sample to predict.
    """

    sample = np.array(arg2)

    prediction = model.predict(sample)

    print(prediction)


# === MAIN ==============================================

if __name__ == "__main__":

    model = import_model("K10")

