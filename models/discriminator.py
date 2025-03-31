import tensorflow as tf
from tensorflow.keras import layers, Model
from .layers import EdgeAttention

class Generator(Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._build_model()

    def _build_model(self):
        inputs = layers.Input((self.config['model']['img_size'], self.config['model']['img_size'], self.config['model']['channels']))
        
        # Encoder
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Bridge
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = EdgeAttention()(x)
        
        # Decoder
        x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        outputs = layers.Conv2D(1, 4, padding='same', activation='sigmoid')(x)
        
        self.model = Model(inputs, outputs, name='Generator')
    
    def call(self, inputs):
        return self.model(inputs)
