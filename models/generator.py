import tensorflow as tf
from tensorflow.keras import layers, Model
from .layers import SpectralNormalization

class Discriminator(Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._build_model()

    def _build_model(self):
        inputs = layers.Input((self.config['model']['img_size'], self.config['model']['img_size'], self.config['model']['channels']))
        
        x = SpectralNormalization(layers.Conv2D(64, 4, strides=2, padding='same'))(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        x = SpectralNormalization(layers.Conv2D(128, 4, strides=2, padding='same'))(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = SpectralNormalization(layers.Conv2D(256, 4, strides=2, padding='same'))(x)
        x = layers.LeakyReLU(0.2)(x)
        
        outputs = layers.Conv2D(1, 4, padding='same')(x)
        
        self.model = Model(inputs, outputs, name='Discriminator')
    
    def call(self, inputs):
        return self.model(inputs)
