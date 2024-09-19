# Crop Weed Detection using Deep Convolutional Networks

This project implements a deep learning model for crop and weed detection using the Plant Seedlings Dataset. It's designed to help in the automated identification of plant species at early growth stages, which is crucial for effective weed control and crop management in agriculture.

## Project Overview

The notebook (`undp_crop_detection_dataset.py`) contains code for:

1. Loading and preprocessing the Plant Seedlings Dataset
2. Splitting the data into training and testing sets
3. Creating a Convolutional Neural Network (CNN) model for image classification
4. Training the model on the dataset
5. Evaluating the model's performance
6. Visualizing the training results

## Dataset

The project uses the [Plant Seedlings Dataset](https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset), which includes images of 12 different plant species at various growth stages. The dataset is split into training and testing sets with a 70-30 ratio.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

You can install the required packages using:

```
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Mount your Google Drive and unzip the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -uq "/content/drive/My Drive/Colab Notebooks/UNDP crop detection/v2-plant-seedlings-dataset.zip" -d "/content/drive/My Drive/Colab Notebooks/UNDP crop detection/plant-seedlings-dataset"
```

2. Run the notebook cells sequentially to:
   - Prepare the training data
   - Create the model
   - Train the model
   - Visualize the results

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

- 3 Convolutional layers with ReLU activation
- Max Pooling and Dropout layers for regularization
- Batch Normalization for improved training stability
- Dense layers for classification

The model is compiled using categorical crossentropy loss and RMSprop optimizer.

## Results

The notebook includes visualizations of:

- Training and validation accuracy over epochs
- Training and validation loss over epochs

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

