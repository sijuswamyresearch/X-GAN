import tensorflow as tf
import numpy as np
import yaml
from data import load_medical_images, split_dataset, prepare_datasets, add_xray_noise
from models import MedicalDenoiser
from training import train

def main():
    # Load config
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load and prepare data
    clean_images = load_medical_images(config['data_paths'], config['img_size'])
    X_train, X_val, X_test = split_dataset(clean_images, config['test_size'], config['val_size'])
    
    X_train_noisy = add_xray_noise(X_train, config['peak_photons'])
    X_val_noisy = add_xray_noise(X_val, config['peak_photons'])
    X_test_noisy = add_xray_noise(X_test, config['peak_photons'])
    
    train_dataset = prepare_datasets(X_train, X_train_noisy, config['batch_size'])
    val_dataset = prepare_datasets(X_val, X_val_noisy, config['batch_size'])
    
    # Initialize model
    denoiser = MedicalDenoiser(config['img_size'], config['checkpoint_dir'])
    denoiser.compile(
        g_optimizer=tf.keras.optimizers.Adam(config['g_lr'], config['beta_1']),
        d_optimizer=tf.keras.optimizers.Adam(config['d_lr'], config['beta_1'])
    )
    
    # Train
    train(denoiser, train_dataset, val_dataset, (X_test_noisy, X_test), config['epochs'])

if __name__ == "__main__":
    main()
