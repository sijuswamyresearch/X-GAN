from tensorflow.keras import layers, Model
from .layers import SpectralNormalization

def build_discriminator(img_size: int = 256) -> Model:
    inputs = layers.Input((img_size, img_size, 1))
    x = SpectralNormalization(layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_uniform'))(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = SpectralNormalization(layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_uniform'))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = SpectralNormalization(layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_uniform'))(x)
    x = layers.LeakyReLU(0.2)(x)
    outputs = layers.Conv2D(1, 4, padding='same', kernel_initializer='glorot_uniform')(x)
    return Model(inputs, outputs, name='Discriminator')
