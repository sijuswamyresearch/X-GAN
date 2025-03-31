import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .augmentations import add_xray_noise, normalize_medical_images

def load_medical_images(data_paths, img_size=256):
    images = []
    for path in data_paths:
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (img_size, img_size))
                        images.append(img)
    return normalize_medical_images(np.array(images))

def load_dataset(config):
    clean_images = load_medical_images(config['data']['paths'], config['model']['img_size'])
    X_train, X_temp = train_test_split(clean_images, test_size=config['data']['test_size'] + config['data']['val_size'])
    X_val, X_test = train_test_split(X_temp, test_size=config['data']['test_size']/(config['data']['test_size']+config['data']['val_size']))
    
    train_ds = tf.data.Dataset.from_tensor_slices((add_xray_noise(X_train), X_train))
    val_ds = tf.data.Dataset.from_tensor_slices((add_xray_noise(X_val), X_val))
    test_ds = tf.data.Dataset.from_tensor_slices((add_xray_noise(X_test), X_test))
    
    return (
        train_ds.shuffle(1000).batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE),
        val_ds.batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE),
        test_ds.batch(config['training']['batch_size'])
    )
