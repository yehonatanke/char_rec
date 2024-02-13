## Structure

1. **data_preparation.py**: Contains functions for organizing the dataset and preprocessing the data.
   - File containing functions for organizing the dataset and preprocessing the data. This includes loading images, resizing, normalization, and tokenizing text labels.

2. **model.py**: Defines the model architecture, including the encoder, decoder, and attention mechanism.
   - File containing the model architecture, including the encoder, decoder, and attention mechanism. This includes the implementation of convolutional layers for feature extraction in the encoder, and LSTM with attention mechanism in the decoder.

3. **training.py**: Provides the training loop and related functions.
   - File containing the training loop and related functions. This includes functions for training the model on the training data, validating it on the validation data, and computing the loss function.

4. **inference.py**: Includes functions for using the trained model to make predictions.
   - File containing functions for using the trained model to make predictions. This includes loading a trained model, predicting text labels for new images, and any other inference-related functionalities.

5. **samaritan_script_understanding.py**: Main script for data preparation, training, and inference.
   - Main script to run for data preparation, training, and inference. This script orchestrates the entire process, including data loading, preprocessing, model training, and inference.

## Description

- `data_preparation.py`: Functions for dataset organization and preprocessing.
- `model.py`: Model architecture definition.
- `training.py`: Training loop and related functions.
- `inference.py`: Prediction functions.
- `samaritan_script_understanding.py`: Main script for executing the program.
