![bluearia logo](https://github.com/FlakeDo/BlueAria/blob/main/assets/logo_horizontal_onwhite.png)

# BlueAria

Whale sounds classification using spctrographic processing via CNN. This project was developed as part of the University of Nice’s FabLab team, in collaboration with Fredy Emmanuel Hounnou. The short-term objective is to train models capable of classifying whale sounds by species with acceptable accuracy. The long-term goal is to create a model that can identify recurrent patterns in the calls of a given species, much like how one would try to figure out the rules of an unknowned langage.

## CNN Models

The models are made using two Python libraries: Keras and Ultralytics YOLO. As this project is my first experience with CNNs and AI in general, my approach is not to immediately use the latest technology but rather to learn step by step.

All the models will be trained using the WhaleSounds dataset available on Hugging Face. This dataset consists of 105,163 underwater acoustic recordings from around Antarctica, manually annotated for seven different types of whale calls, resampled to a consistent sampling frequency of 250 Hz.


## Data Processing

Data within the dataset are imported and processed in two different ways, one for each model type. The first is the more classic approach, processing NumPy files from within the dataset to generate spectrograms and create training data. All the tools for data processing are located in data.py.

The second approach is required to adapt this data for YOLO usage. YOLO models require YAML files for training, which is why yolo_data.py exists.
