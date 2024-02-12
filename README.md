



# Samaritan Script Understanding

## Overview
This project aims to develop a deep learning model for understanding text written in the Samaritan script using image processing techniques and sequence-to-sequence models with attention mechanisms. The model is trained on a dataset containing images of Samaritan writings along with their corresponding text labels.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- PIL (Python Imaging Library)
- scikit-learn (for text preprocessing)

## Dataset
The dataset used for training the model consists of images containing Samaritan writings and their corresponding text labels. The dataset should be organized into separate directories for images and text files, with each image file paired with its corresponding text file.

**Example directory structure:**

```bash
samaritan_data/
│
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── texts/
    ├── text1.txt
    ├── text2.txt
    └── ...

```

## Mathematics of Sequence-to-Sequence Model with Attention

### Encoder
The encoder processes the input image to obtain a fixed-size representation.

1. **Input Image**: Let $\( x \)$ denote the input image with dimensions $\( H \times W \times C \)$, where $\( H \)$ is the height, $\( W \)$ is the width, and $\( C \)$ is the number of channels.
2. **Convolutional Layers**: The encoder consists of convolutional layers to extract features from the input image.
    - Let $\( f_{enc} \)$ represent the encoder function, which outputs a feature map $\( V \)$ of size $\( H' \times W' \times D \)$, where $\( D \)$ is the depth of the feature map.
3. **Flatten Operation**: The feature map $\( V \)$ is flattened into a vector $\( v \)$ of size $\( N \times D \)$, where $\( N = H' \times W' \)$.

### Decoder with Attention
The decoder generates the output text sequence while attending to relevant parts of the input image.

1. **Input Text Sequence**: Let $y=(y_1,y_2, ...,y_T)$ denote the target text sequence, where $\( T \)$ is the maximum sequence length.
2. **Embedding Layer**: The target text sequence $\( y \)$ is embedded into a continuous representation.
    - Let $\( E \)$ be the embedding matrix.
    - The embedded sequence $y=(y_1,y_2, ...,y_T)$ has dimensions $\( T \times D' \)$, where $\( D' \)$ is the embedding dimension.
3. **Decoder LSTM with Attention**: The decoder LSTM attends to relevant parts of the input image while generating the output text sequence.
    - Let $\( h_t \)$ denote the hidden state of the LSTM at time step $\( t \)$.
    - The attention mechanism computes context vectors $\( c_t \)$ based on the encoder features and the decoder hidden state.
    - The decoder LSTM outputs probability distributions over the vocabulary at each time step.
    - Let $\( p_t \)$ represent the output probability distribution at time step $\( t \)$.
4. **Loss Function**: The loss function measures the discrepancy between the predicted probability distribution $\( p_t \)$ and the true next token in the target sequence $\( y_{t+1} \)$.

### Training Objective
The model is trained to minimize the negative log likelihood of the target sequence given the input image:

$$\mathcal{L}=-\sum_{t=1}^{T} \\log p_t(y_t | x, y_{<t>})\$$

where $y_{t'}$ ( for $t'$ < $t$ ) denotes the previously generated tokens.

## Usage
1. **Data Preparation**: Organize your dataset as described above.
2. **Install Dependencies**: Install the required Python packages listed in the requirements section.
3. **Preprocess Data**: Implement data loading and preprocessing functions in the `load_and_preprocess_dataset()` method in the main script (`samaritan_script_understanding.py`). This includes loading images, preprocessing images (resize, normalization), tokenizing text labels, and converting them to numerical sequences.
4. **Model Training**: Run the main script (`samaritan_script_understanding.py`) to train the model. The script will load and preprocess the dataset, create and compile the model, train the model on the training data, and validate it on the validation data. Adjust hyperparameters, model architecture, and training settings as needed.
5. **Inference**: Once the model is trained, you can use it to predict text labels for new images by loading the trained model and passing the images through the model.

## Model Architecture
The model architecture consists of an image encoder (CNN) and a text decoder (RNN with attention mechanism). The image encoder extracts features from input images, while the text decoder generates text sequences based on the encoded image features. The attention mechanism allows the model to focus on relevant parts of the input image when generating the output text.

## Contributing
Contributions to this project are welcome. If you have any ideas, suggestions, or improvements, feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

