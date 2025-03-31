import tensorflow as tf
import numpy as np
import datetime
from typing import Dict, Tuple
from utils.logger import setup_logger
from utils.metrics import calculate_epi
from utils.visualize import plot_loss_curves

def train(denoiser, train_dataset, val_dataset, test_data, epochs=150):
    logger = setup_logger()
    history = {'train': {'d_loss': [], 'g_loss': []}, 'val': {'d_loss': [], 'g_loss': []}}
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training loop
        for batch, (noisy, clean) in enumerate(train_dataset):
            losses = denoiser.train_step((noisy, clean))
            if batch % 50 == 0:
                logger.info(f"Batch {batch}: D Loss: {losses['d_loss']:.4f}, G Loss: {losses['g_loss']:.4f}")
        
        # Validation loop
        for noisy, clean in val_dataset:
            val_losses = denoiser.validation_step((noisy, clean))
        
        # Update history
        for k in losses:
            history['train'][k].append(losses[k].numpy())
            history['val'][k].append(val_losses[k].numpy())
    
    # Plot and save results
    plot_loss_curves(history, epochs)
    return history
