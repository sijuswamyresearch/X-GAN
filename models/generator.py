import tensorflow as tf
from tensorflow.keras import layers, Model
from models.layers import EdgeAttention

class Generator(Model):
    def __init__(self, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.build_model()
        
    def build_model(self):
        inputs = layers.Input((self.img_size, self.img_size, 1))
        
        # Encoder
        d1 = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        d1 = layers.LeakyReLU(0.2)(d1)
        
        d2 = layers.Conv2D(128, 4, strides=2, padding='same')(d1)
        d2 = layers.BatchNormalization()(d2)
        d2 = layers.LeakyReLU(0.2)(d2)
        
        # Bridge with attention
        bridge = layers.Conv2D(256, 4, strides=2, padding='same')(d2)
        bridge = layers.BatchNormalization()(bridge)
        bridge = EdgeAttention()(bridge)
        
        # Decoder
        u1 = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(bridge)
        u1 = layers.BatchNormalization()(u1)
        u1 = layers.Concatenate()([u1, d2])
        u1 = layers.ReLU()(u1)
        
        u2 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(u1)
        u2 = layers.BatchNormalization()(u2)
        u2 = layers.Concatenate()([u2, d1])
        u2 = layers.ReLU()(u2)
        
        outputs = layers.Conv2D(1, 4, padding='same', activation='sigmoid')(u2)
        
        self.model = Model(inputs, outputs, name='Generator')
    
    def call(self, inputs):
        return self.model(inputs)
