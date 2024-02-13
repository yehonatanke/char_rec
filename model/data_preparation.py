# data_preparation.py

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

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
def preprocess_data(images, texts):
    # Implement preprocessing steps such as resizing images, tokenizing text labels, etc.
    pass

# Function to split the dataset into training and validation sets
def split_dataset(images, texts, test_size=0.2):
    return train_test_split(images, texts, test_size=test_size, random_state=42)
