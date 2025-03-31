import tensorflow as tf
from datetime import datetime
from utils.logger import setup_logger
from utils.metrics import calculate_epi
from utils.visualize import plot_loss_curves

class XGANTrainer:
    def __init__(self, model, train_ds, val_ds, config):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.config = config
        self.logger = setup_logger()
        
    def train_step(self, batch):
        noisy, clean = batch
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass and loss calculations
            losses = self.model.compute_losses(clean, noisy)
        
        # Apply gradients
        self.model.apply_gradients(losses, tape)
        del tape
        return losses
    
    def train(self, epochs):
        history = {'train': [], 'val': []}
        
        for epoch in range(epochs):
            train_losses = []
            for batch in self.train_ds:
                losses = self.train_step(batch)
                train_losses.append(losses)
            
            # Validation and logging
            val_losses = self.validate()
            self.log_epoch(epoch, train_losses, val_losses)
            history['train'].append(train_losses)
            history['val'].append(val_losses)
            
        plot_loss_curves(history)
        return history
