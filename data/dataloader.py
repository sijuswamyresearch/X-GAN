import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import List, Tuple

def load_medical_images(data_paths: List[str], img_size: int = 256) -> np.ndarray:
    """Loads, preprocesses, and returns a dataset of medical images."""
    images = []
    for path in data_paths:
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if not os.path.isdir(folder_path):
                continue
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (img_size, img_size))
                        images.append(img)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)
    return images

def split_dataset(images: np.ndarray, test_size: float = 0.1, val_size: float = 0.1) -> Tuple:
    X_train, X_temp = train_test_split(images, test_size=(test_size + val_size))
    X_val, X_test = train_test_split(X_temp, test_size=(test_size / (test_size + val_size)))
    return X_train, X_val, X_test

def prepare_datasets(clean_images: np.ndarray, noisy_images: np.ndarray, batch_size: int = 16) -> tf.data.Dataset:
    def augment(noisy, clean):
        if tf.random.uniform(()) > 0.5:
            noisy = tf.image.flip_left_right(noisy)
            clean = tf.image.flip_left_right(clean)
        if tf.random.uniform(()) > 0.5:
            noisy = tf.image.flip_up_down(noisy)
            clean = tf.image.flip_up_down(clean)
        return noisy, clean
    
    return (tf.data.Dataset.from_tensor_slices((noisy_images, clean_images))
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(1000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
