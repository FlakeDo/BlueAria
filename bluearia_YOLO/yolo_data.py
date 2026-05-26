from __future__ import annotations

import os
import sys
from pathlib import Path
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the path of the parent folder (BlueAria/) to the path, to be able to import data

from data import load_training_materials


def create_yaml_dataset_dir(dir_name: str) -> dict:
    """Create dataset directory structure for YOLO. In the end, the directory yaml_dir_name will contain the dataset
    as a yaml file and two sub-directories containting images and labels, both for training and validation.

    Args:
        dir_name (str): Name for the directory.

    Returns:
        dict: A dictionnary containing all the path to train and tests files.
    """

    root = f"yaml_{dir_name}"

    dirs = {
        "root": root,
        "images_train": os.path.join(root, "images/train"),
        "images_val": os.path.join(root, "images/val"),
        "labels_train": os.path.join(root, "labels/train"),
        "labels_val": os.path.join(root, "labels/val"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)   # Exist_ok is set to true to avoid raising an OS error.

    return dirs

def save_spectrogram_image(spec: np.ndarray, path: str | Path) -> None:
    """Save spectrogram as image. YOLO only works on images, so the data has to be converted to images.
    This is done using matplotlib. The spectrogram given as a parameter will be stored as an image 
    where the path is leading.

    Args:
        spec (np.ndarray): The spectrogram to save.
        path (str | Path): The path where the image will be saved.
    """
    plt.figure(figsize=(2,2))
    plt.imshow(spec[:,:,0], aspect="auto", origin="lower")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

def create_yolo_label_file(label: int, path: str | Path) -> None:
    """Create YOLO label file. YOLO's format is <class id> x_center, y_center width height, hence the four floats
    written in the text file.

    Args:
        label (int): The current class for which the file is created.
        path (str | Path): Where the file will be saved.
    """
    with open(path, "w") as f:
        f.write(f"{label} 0.5 0.5 1.0 1.0\n")

def convert_dataset_to_yolo(X: np.ndarray, y: np.ndarray, dirs: dict, split: str="train") -> None:
    """Convert spectrogram dataset to YOLO format. If split is set to train, everything will be saved in train directories,
    otherwise it will end up in the two val dirs, used for other purpose.

    Args:
        X (np.ndarray): An array containing all the samples.
        Y (np.ndarray): An array containing all the labels related to the samples.
        dirs (dict): A dictionary containing all the paths leading to yolo files directories.
        split (str): An argument used to chose if the dataset is stored in the training directory, or other for testing, val directory.
    """

    if split == "train":
        img_dir = dirs["images_train"]
        label_dir = dirs["labels_train"]
    else:
        img_dir = dirs["images_val"]
        label_dir = dirs["labels_val"]

    labels = np.argmax(y, axis=1) # Returns the highest label in the list
    nb_samples = len(X)

    for i in range(nb_samples):

        img_path = os.path.join(img_dir, f"spec_{i}.png") # Get the path where the image/lebel will be saved
        label_path = os.path.join(label_dir, f"spec_{i}.txt")

        save_spectrogram_image(X[i], img_path)      # Will take every signal and save them in the specified dir as a png spectrogram.
        create_yolo_label_file(labels[i], label_path)   # Will add a new text file containing the YOLO format for each signal.

        sys.stdout.write("Progress [%d%%]\r" % round(i/nb_samples * 100))   # Display the progression of the operation.
        sys.stdout.flush()

def create_dataset_yaml(dirs: dict, class_names: list) -> None:
    """Create YOLO dataset's yaml file. Contains pretty much every useful informations about the dataset. 

    Args:
        dirs (dict): A dictionary containing all the paths leading to yolo files directories.
        class_names (list): A list of the names of classes as strings.
    """

    yaml_data = {
        "path" : dirs['root'],
        "train" : "images/train",
        "val" : "images/val",
        "names" : class_names,
        "nc" : len(class_names)
    }

    yaml_path = os.path.join(dirs["root"], "dataset.yaml")

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    print(f"YAML dataset successfully created at {yaml_path}")




def export_training_materials_YOLO(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, dataset_name: str) -> None:
    """The main function of the set of conversion tool. It creates directories for the materials to be saved in, converts the 
    dataset to something YOLO can use (both for training and other usage), and finally creates the dataset's yaml file.
    Class_names contains 7 classes : the original dataset contains 8 classes, but the 8th one contains unspecified
    whale calls, useless for model training (but used to test the model afterward).

    Args:
        X_train (np.ndarray): The samples used for training.
        y_train (np.ndarray): The labels associated with the samples used for training.
        X_test (np.ndarray): The samples used for testing.
        y_test (np.ndarray): The labels associated with the samples used for testing.
        dataset_name (str): The name for the dataset, used to label the directory.
    """

    dirs = create_yaml_dataset_dir(dataset_name)

    class_names = [
        "class_0",
        "class_1",
        "class_2",
        "class_3",
        "class_4",
        "class_5",
        "class_6"
    ]

    convert_dataset_to_yolo(X_train, y_train, dirs, "train")
    convert_dataset_to_yolo(X_test, y_test, dirs, "val")

    create_dataset_yaml(dirs,class_names)

    print("YOLO dataset successfully generated !")


def create_data_YOLO(data_dir: str | Path="data_default") -> None:
    """Create YOLO usable materials. Might take a long time depending on the dataset. (it took 4h for the 
    WhaleSounds_HuggingFace dataset.)

    Args:
        data_dir (str | Path): The path where the data to convert is located.
    """

    X_train, y_train, X_test, y_test = load_training_materials(data_dir)  # Choose the type of data to work with.

    export_training_materials_YOLO(X_train, y_train, X_test, y_test, "WhaleSounds_HuggingFace_dataset")


def get_YOLO_data(data_path: str | Path) -> str:
    """Returns the name of yaml file associated with the specified dataset directory as a string.

    Args:
        data_path (str | Path): The directory where the yolo dataset is stored.
    
    Returns:
        str: The path to the yaml file, as a string.
    
    Raises:
        Exception: Yaml file not found.
    """

    try:
        yaml_file = glob.glob(os.path.join(data_path,'*.yaml'))[0]  # The first file with a yaml extension.
        return yaml_file
    except:
        raise Exception("No yaml file in this directory.")
