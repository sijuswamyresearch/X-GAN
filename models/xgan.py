import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import Model
from .generator import build_generator
from .discriminator import build_discriminator
from .layers import SobelEdgeLayer
from training.losses import compute_loss, gradient_penalty

class MedicalDenoiser(Model):
    def __init__(self, img_size=256, checkpoint_dir='training_checkpoints'):
        super().__init__()
        self.img_size = img_size
        self.sobel_edge_layer = SobelEdgeLayer()
        self.generator = build_generator(img_size)
        self.discriminator = build_discriminator(img_size)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator
        )
        self._restore_checkpoint()

    def _restore_checkpoint(self):
        if tf.train.latest_checkpoint(self.checkpoint_dir):
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, batch):
        noisy, real = batch
        with tf.GradientTape(persistent=True) as tape:
            generated = self.generator(noisy, training=True)
            real_logits = self.discriminator(real, training=True)
            fake_logits = self.discriminator(generated, training=True)
            
            gp = gradient_penalty(self.discriminator, real, generated)
            d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits) + 10.0 * gp
            
            recon_loss = compute_loss(real, generated, self.sobel_edge_layer)
            adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_logits), fake_logits, from_logits=True))
            g_loss = recon_loss + 0.1 * adv_loss

        d_grad = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grad = tape.gradient(g_loss, self.generator.trainable_variables)
        
        self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))
        
        return {'d_loss': d_loss, 'g_loss': g_loss}
