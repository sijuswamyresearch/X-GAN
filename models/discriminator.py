import tensorflow as tf
from tensorflow.keras import layers, Model
from models.layers import SpectralNormalization

class Discriminator(Model):
    def __init__(self, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.build_model()
        
    def build_model(self):
        inputs = layers.Input((self.img_size, self.img_size, 1))
        
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
