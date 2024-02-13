import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load images and corresponding texts from the dataset directory
def load_images_and_texts(data_dir):
    images_dir = os.path.join(data_dir, 'images')
    texts_dir = os.path.join(data_dir, 'texts')
    image_files = sorted(os.listdir(images_dir))
    text_files = sorted(os.listdir(texts_dir))
    images = []
    texts = []
    for img_file, txt_file in zip(image_files, text_files):
        img_path = os.path.join(images_dir, img_file)
        with Image.open(img_path) as img:
            images.append(np.array(img))
        txt_path = os.path.join(texts_dir, txt_file)
        with open(txt_path, 'r') as txt:
            texts.append(txt.read().strip())
    return images, texts

# Function to preprocess the loaded images and texts
def preprocess_data(images, texts, image_size=(224, 224), max_text_length=100):
    """
    Preprocesses the loaded images and texts.

    Parameters:
        images (list): List of image arrays.
        texts (list): List of text strings.
        image_size (tuple): Target size for resizing the images.
        max_text_length (int): Maximum length of the text sequences.

    Returns:
        preprocessed_images (numpy array): Preprocessed image data.
        preprocessed_texts (numpy array): Preprocessed text data.
    """
    # Resize images
    preprocessed_images = np.array([image.img_to_array(image.array_to_img(img).resize(image_size)) / 255.0 for img in images])
    
    # Tokenize text labels
    count_vectorizer = CountVectorizer()
    text_counts = count_vectorizer.fit_transform(texts)
    
    # Convert text labels to numerical sequences
    label_encoder = LabelEncoder()
    integer_encoded_texts = label_encoder.fit_transform(texts)
    
    # One-hot encode numerical sequences
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_texts = integer_encoded_texts.reshape(len(integer_encoded_texts), 1)
    onehot_encoded_texts = onehot_encoder.fit_transform(integer_encoded_texts)
    
    # Pad text sequences to max_text_length
    padded_texts = pad_sequences(text_counts.toarray(), maxlen=max_text_length, padding='post')
    
    return preprocessed_images, padded_texts

# Function to split the dataset into training and validation sets
def split_dataset(images, texts, test_size=0.2):
    return train_test_split(images, texts, test_size=test_size, random_state=42)
