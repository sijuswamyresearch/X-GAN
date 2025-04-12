## This version of `Python` code is for Kaggle users. They can create a Kaggle notebook and paste this code and set the path properly after uploading the datasets.

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, initializers
from sklearn.model_selection import KFold, train_test_split
import cv2
import datetime
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import logging
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# --- Constants and Configurations ---
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
PEAK_PHOTONS = 1000
TEST_SIZE = 0.1
VAL_SIZE = 0.1
TRAIN_SIZE = 1.0 - TEST_SIZE - VAL_SIZE
DATA_PATHS = [<your data path> ]
LOG_DIR = "logs"
MODEL_DIR = "models"
RESULTS_DIR = "results"
CHECKPOINT_DIR = 'training_checkpoints'
KFOLDS = 5  # Number of folds for cross-validation

# --- Logging Setup ---
def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(LOG_DIR)

# --- Utility Functions ---
def normalize_medical_images(images: np.ndarray) -> np.ndarray:
    """ Normalizes images within the 0-1 range with percentile clipping."""
    try:
        p_min = np.percentile(images, 0.1)
        p_max = np.percentile(images, 99.9)
        normalized = np.clip(images, p_min, p_max)
        normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized) + 1e-7)
    except Exception as e:
        logger.error(f"Error in image normalization: {e}")
        raise
    if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
        logger.error("Normalized images contain NaN or Inf values.")
        raise ValueError("Normalized images contain NaN or Inf values.")
    return normalized.astype(np.float32)

def add_xray_noise(images: np.ndarray, peak: int = PEAK_PHOTONS) -> np.ndarray:
    """Adds Poisson noise to simulate X-ray imaging."""
    try:
        scaled = images * peak
        noisy = np.random.poisson(scaled).astype(np.float32) / peak
        noisy = np.clip(noisy, 0, 1)
    except Exception as e:
        logger.error(f"Error in adding X-ray noise: {e}")
        raise
    return noisy

def load_medical_images(data_paths: List[str], img_size: int = IMG_SIZE) -> np.ndarray:
    """Loads, preprocesses, and returns a dataset of medical images."""
    images = []
    total_files = 0
    loaded_files = 0
    for path in data_paths:
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if not os.path.isdir(folder_path):
                continue
            for filename in os.listdir(folder_path):
                total_files += 1
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, filename)
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            logger.warning(f"Failed to load image (None) at: {img_path}")
                            continue
                        img = cv2.resize(img, (img_size, img_size))
                        images.append(img)
                        loaded_files += 1
                    except Exception as e:
                        logger.error(f"Error loading image at {img_path}: {e}")
                        continue
    images = np.array(images)
    if images.size == 0:
        logger.error(f"No images found in the provided data paths: {data_paths}")
        raise ValueError(f"No images found in the provided data paths: {data_paths}")
    images = normalize_medical_images(images)
    images = np.expand_dims(images, axis=-1)  # Adding channel dimension
    logger.info(f"Loaded {loaded_files} images out of {total_files} from {data_paths}.")
    return images

def prepare_datasets(clean_images: np.ndarray, noisy_images: np.ndarray, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Creates TensorFlow datasets for training."""
    def augment(noisy: tf.Tensor, clean: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Augments image with random flips."""
        if tf.random.uniform(()) > 0.5:
            noisy = tf.image.flip_left_right(noisy)
            clean = tf.image.flip_left_right(clean)
        if tf.random.uniform(()) > 0.5:
            noisy = tf.image.flip_up_down(noisy)
            clean = tf.image.flip_up_down(clean)
        return tf.cast(noisy, tf.float32), tf.cast(clean, tf.float32)
    try:
        dataset = (tf.data.Dataset.from_tensor_slices((noisy_images, clean_images))
                  .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                  .shuffle(1000)
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE))
    except Exception as e:
        logger.error(f"Error in creating TF Dataset: {e}")
        raise
    return dataset

def create_noisy_dataset(images: np.ndarray) -> np.ndarray:
    noisy_images = add_xray_noise(images)
    return noisy_images

def generate_synthetic_data(num_samples: int = 1000, img_size: int = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic images with noise."""
    clean_images = np.random.uniform(0, 1, (num_samples, img_size, img_size, 1)).astype(np.float32)
    noisy_images = add_xray_noise(clean_images)
    return clean_images, noisy_images

# --- Custom Layers ---
class SobelEdgeLayer(layers.Layer):
    """Custom layer to compute Sobel edge magnitude."""
    def __init__(self, **kwargs):
        super(SobelEdgeLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=-1)
        sobel = tf.image.sobel_edges(inputs)
        sobel_mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1) + 1e-7)
        return sobel_mag

class SqueezeLayer(layers.Layer):
    """Custom layer to remove a dimension from the tensor."""
    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(inputs, axis=self.axis)

class SpectralNormalization(layers.Wrapper):
    """Layer wrapper to perform spectral normalization."""
    def __init__(self, layer: layers.Layer, iteration: int = 1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.iteration = iteration

    def build(self, input_shape: tf.TensorShape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.in_dim = tf.reduce_prod(self.w_shape[:-1])
        self.out_dim = self.w_shape[-1]
        self.u = self.add_weight(shape=(1, self.out_dim), initializer='random_normal', trainable=False, name='sn_u', dtype=tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        self._compute_weights()
        return self.layer(inputs)

    def _compute_weights(self):
        w_reshaped = tf.reshape(self.w, [self.in_dim, self.out_dim])
        u = self.u
        for _ in range(self.iteration):
            v = tf.matmul(u, w_reshaped, transpose_b=True)
            v = tf.math.l2_normalize(v, epsilon=1e-12)
            u = tf.matmul(v, w_reshaped)
            u = tf.math.l2_normalize(u, epsilon=1e-12)
        sigma = tf.matmul(tf.matmul(u, w_reshaped, transpose_b=True), v, transpose_b=True)
        w_normalized = self.w / sigma
        self.layer.kernel.assign(w_normalized)
        self.u.assign(u)

class EdgeAttention(layers.Layer):
    """Custom layer for Edge Attention."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sobel_edge_layer = SobelEdgeLayer()
        self.conv = layers.Conv2D(1, 1, activation='sigmoid', kernel_initializer='he_uniform')
        self.multiply = layers.Multiply()
        self.input_channels = None

    def build(self, input_shape: tf.TensorShape):
        self.input_channels = input_shape[-1]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        sobel_mag = self.sobel_edge_layer(x)
        att = self.conv(sobel_mag)
        att = tf.tile(att, multiples=[1, 1, 1, self.input_channels])
        return self.multiply([x, att])

# --- Edge Preservation Index Calculation ---
def calculate_epi(original: np.ndarray, denoised: np.ndarray, window_size: int = 5) -> float:
    """
    Computes the Edge Preservation Index (EPI) between original and denoised images.
    
    Args:
        original: Original clean image (2D array)
        denoised: Denoised image (2D array)
        window_size: Size of the window for local statistics computation
        
    Returns:
        EPI value (higher is better, max 1.0)
    """
    # Ensure images are 2D and float32
    original = original.squeeze().astype(np.float32)
    denoised = denoised.squeeze().astype(np.float32)
    
    # Compute gradients using Sobel operator
    grad_x_orig = cv2.Sobel(original, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_orig = cv2.Sobel(original, cv2.CV_32F, 0, 1, ksize=3)
    grad_orig = np.sqrt(grad_x_orig**2 + grad_y_orig**2)
    
    grad_x_den = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_den = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    grad_den = np.sqrt(grad_x_den**2 + grad_y_den**2)
    
    # Normalize gradients to [0, 1]
    grad_orig = (grad_orig - grad_orig.min()) / (grad_orig.max() - grad_orig.min() + 1e-7)
    grad_den = (grad_den - grad_den.min()) / (grad_den.max() - grad_den.min() + 1e-7)
    
    # Pad images for window processing
    pad = window_size // 2
    grad_orig_pad = np.pad(grad_orig, pad, mode='reflect')
    grad_den_pad = np.pad(grad_den, pad, mode='reflect')
    
    epi_values = []
    for i in range(pad, grad_orig.shape[0] + pad):
        for j in range(pad, grad_orig.shape[1] + pad):
            # Extract windows
            window_orig = grad_orig_pad[i-pad:i+pad+1, j-pad:j+pad+1]
            window_den = grad_den_pad[i-pad:i+pad+1, j-pad:j+pad+1]
            
            # Compute means
            mean_orig = np.mean(window_orig)
            mean_den = np.mean(window_den)
            
            # Compute numerator and denominator
            numerator = np.sum((window_orig - mean_orig) * (window_den - mean_den))
            denom_orig = np.sum((window_orig - mean_orig)**2)
            denom_den = np.sum((window_den - mean_den)**2)
            denominator = np.sqrt(denom_orig * denom_den)
            
            if denominator > 1e-7:  # Avoid division by zero
                epi = numerator / denominator
                epi_values.append(epi)
    
    return np.mean(epi_values) if epi_values else 0.0

# --- Model Definition ---
class MedicalDenoiser(tf.keras.Model):
    """Medical Image Denoiser GAN Model."""
    def __init__(self, img_size: int = IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.sobel_edge_layer = SobelEdgeLayer()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.edge_weight = 5.0
        self.g_optimizer = tf.keras.optimizers.Adam(5e-6, beta_1=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(2e-6, beta_1=0.9)
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.g_optimizer,
            discriminator_optimizer=self.d_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        self._restore_checkpoint()

    def _restore_checkpoint(self) -> None:
        """Restores from a checkpoint."""
        try:
            if tf.train.latest_checkpoint(self.checkpoint_dir):
                self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
                logger.info(f"Restored checkpoint from: {tf.train.latest_checkpoint(self.checkpoint_dir)}")
            else:
                logger.info("No checkpoint found. Starting from scratch.")
        except Exception as e:
            logger.error(f"Error during checkpoint restoration: {e}")

    def build_generator(self) -> Model:
        """Builds the generator network."""
        inputs = layers.Input((self.img_size, self.img_size, 1), dtype=tf.float32)
        d1 = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_uniform')(inputs)
        d1 = layers.LeakyReLU(0.2)(d1)
        d2 = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_uniform')(d1)
        d2 = layers.BatchNormalization()(d2)
        d2 = layers.LeakyReLU(0.2)(d2)
        bridge = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_uniform')(d2)
        bridge = layers.BatchNormalization()(bridge)
        bridge = EdgeAttention()(bridge)
        u1 = layers.Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer='he_uniform')(bridge)
        u1 = layers.BatchNormalization()(u1)
        u1 = layers.Concatenate()([u1, d2])
        u1 = layers.ReLU()(u1)
        u2 = layers.Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer='he_uniform')(u1)
        u2 = layers.BatchNormalization()(u2)
        u2 = layers.Concatenate()([u2, d1])
        u2 = layers.ReLU()(u2)
        u3 = layers.Conv2DTranspose(32, 4, strides=2, padding='same', kernel_initializer='he_uniform')(u2)
        u3 = layers.BatchNormalization()(u3)
        u3 = layers.ReLU()(u3)
        outputs = layers.Conv2D(1, 4, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')(u3)
        return Model(inputs, outputs, name='Generator')

    def build_discriminator(self) -> Model:
        """Builds the discriminator network."""
        inputs = layers.Input((self.img_size, self.img_size, 1), dtype=tf.float32)
        x = SpectralNormalization(layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_uniform'))(inputs)
        x = layers.LeakyReLU(0.2)(x)
        x = SpectralNormalization(layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_uniform'))(x)
        x = layers.LeakyReLU(0.2)(x)
        x = SpectralNormalization(layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_uniform'))(x)
        x = layers.LeakyReLU(0.2)(x)
        outputs = layers.Conv2D(1, 4, padding='same', kernel_initializer='glorot_uniform')(x)
        return Model(inputs, outputs, name='Discriminator')

    def gradient_penalty(self, real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
        """Compute the gradient penalty for WGAN-GP."""
        alpha = tf.random.uniform(shape=[tf.shape(real)[0], 1, 1, 1])
        interpolated = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-7)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        tf.debugging.assert_all_finite(gp, "Gradient penalty contains NaN or Inf.")
        return gp

    def compute_loss(self, real: tf.Tensor, generated: tf.Tensor) -> tf.Tensor:
        """Compute the total loss."""
        mae = tf.reduce_mean(tf.abs(real - generated))
        real_edges = self.sobel_edge_layer(real)
        gen_edges = self.sobel_edge_layer(generated)
        edge_loss = tf.reduce_mean(tf.abs(real_edges - gen_edges))
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(real, generated, max_val=1.0))

        total_loss = (
            1.0 * mae +
            5.0 * edge_loss +
            2.0 * ssim_loss
        )
        tf.debugging.assert_all_finite(mae, "MAE contains NaN or Inf.")
        tf.debugging.assert_all_finite(edge_loss, "Edge loss contains NaN or Inf.")
        tf.debugging.assert_all_finite(ssim_loss, "SSIM loss contains NaN or Inf.")
        tf.debugging.assert_all_finite(total_loss, "Total loss contains NaN or Inf.")
        return total_loss

    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Dict:
        """Performs one training step for the GAN."""
        noisy, real = inputs
        noisy = tf.cast(noisy, tf.float32)
        real = tf.cast(real, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            generated = self.generator(noisy, training=True)
            tf.debugging.assert_all_finite(generated, "Generator output contains NaN or Inf.")
            real_logits = self.discriminator(real, training=True)
            fake_logits = self.discriminator(generated, training=True)
            tf.debugging.assert_all_finite(real_logits, "Real logits contain NaN or Inf.")
            tf.debugging.assert_all_finite(fake_logits, "Fake logits contain NaN or Inf.")
            gp = self.gradient_penalty(real, generated)
            lambda_gp = 10.0
            d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits) + lambda_gp * gp

            recon_loss = self.compute_loss(real, generated)
            adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_logits), fake_logits, from_logits=True))
            g_loss = recon_loss + 0.1 * adv_loss  # You can use WGAN loss here
            tf.debugging.assert_all_finite(d_loss, "Discriminator loss contains NaN or Inf.")
            tf.debugging.assert_all_finite(g_loss, "Generator loss contains NaN or Inf.")

        d_grad = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grad = tape.gradient(g_loss, self.generator.trainable_variables)

        for grad, var in zip(d_grad, self.discriminator.trainable_variables):
            tf.debugging.assert_all_finite(grad, f"Discriminator gradient for {var.name} contains NaN or Inf.")
        for grad, var in zip(g_grad, self.generator.trainable_variables):
            tf.debugging.assert_all_finite(grad, f"Generator gradient for {var.name} contains NaN or Inf.")

        d_grad = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None for grad in d_grad]
        g_grad = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None for grad in g_grad]
        self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))
        del tape
        return {'d_loss': d_loss, 'g_loss': g_loss, 'mae': tf.reduce_mean(tf.abs(real - generated))}

    @tf.function
    def validation_step(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Dict:
        """Performs one validation step for the GAN, without training the parameters"""
        noisy, real = inputs
        noisy = tf.cast(noisy, tf.float32)
        real = tf.cast(real, tf.float32)

        generated = self.generator(noisy, training=False)
        tf.debugging.assert_all_finite(generated, "Generator output contains NaN or Inf.")

        real_logits = self.discriminator(real, training=False)
        fake_logits = self.discriminator(generated, training=False)
        tf.debugging.assert_all_finite(real_logits, "Real logits contain NaN or Inf.")
        tf.debugging.assert_all_finite(fake_logits, "Fake logits contain NaN or Inf.")
        gp = self.gradient_penalty(real, generated)
        lambda_gp = 10.0
        d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits) + lambda_gp * gp

        recon_loss = self.compute_loss(real, generated)
        adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_logits), fake_logits, from_logits=True))
        g_loss = recon_loss + 0.1 * adv_loss

        return {'d_loss': d_loss, 'g_loss': g_loss, 'mae': tf.reduce_mean(tf.abs(real - generated))}

    def visualize_results(self, test_noisy: np.ndarray, test_clean: np.ndarray, epoch: int = 0, save: bool = True) -> None:
        """Visualizes denoising results and prints a table of average metrics including EPI."""
        try:
            # Generate denoised images for the first 5 examples
            denoised = self.generator.predict(test_noisy[:5])
            psnr_values = [psnr(c.squeeze(), d.squeeze(), data_range=1.0) for c, d in zip(test_clean[:5], denoised)]
            ssim_values = [ssim(c.squeeze(), d.squeeze(), data_range=1.0) for c, d in zip(test_clean[:5], denoised)]
            
            # Edge metrics
            real_edges = self.sobel_edge_layer(test_clean[:5]).numpy()
            denoised_edges = self.sobel_edge_layer(denoised).numpy()
            edge_rmse_values = [
                np.sqrt(np.mean(np.square(real_edges[i] - denoised_edges[i])))
                for i in range(len(test_clean[:5]))
            ]
            
            # Edge Preservation Index
            epi_values = [
                calculate_epi(test_clean[i], denoised[i])
                for i in range(len(test_clean[:5]))
            ]

            # Visualization: Plot images along with their metrics
            plt.figure(figsize=(20, 15))
            titles = ['Original', 'Noisy', 'Denoised', 'Edge Comparison']
            for i in range(5):
                plt.subplot(5, 4, i * 4 + 1)
                plt.imshow(test_clean[i].squeeze(), cmap='gray')
                plt.title(f"{titles[0]}\nPSNR: {psnr_values[i]:.2f}")
                plt.axis('off')
                plt.subplot(5, 4, i * 4 + 2)
                plt.imshow(test_noisy[i].squeeze(), cmap='gray')
                plt.title(f"{titles[1]}")
                plt.axis('off')
                plt.subplot(5, 4, i * 4 + 3)
                plt.imshow(denoised[i].squeeze(), cmap='gray')
                plt.title(f"{titles[2]}\nSSIM: {ssim_values[i]:.2f}")
                plt.axis('off')
                plt.subplot(5, 4, i * 4 + 4)
                plt.imshow(np.abs(real_edges[i].squeeze() - denoised_edges[i].squeeze()), cmap='gray')
                plt.title(f"{titles[3]}\nEdge RMSE: {edge_rmse_values[i]:.4f}\nEPI: {epi_values[i]:.4f}")
                plt.axis('off')
            
            if save:
                os.makedirs(RESULTS_DIR, exist_ok=True)
                plt.savefig(f'{RESULTS_DIR}/epoch_{epoch}.png')
            plt.close()

            # Console table output with EPI
            header = "{:<10} {:<10} {:<10} {:<15} {:<10}".format("Image", "PSNR", "SSIM", "Edge RMSE", "EPI")
            logger.info(header)
            for i in range(5):
                row = "{:<10} {:<10.2f} {:<10.2f} {:<15.4f} {:<10.4f}".format(
                    i + 1, psnr_values[i], ssim_values[i], edge_rmse_values[i], epi_values[i])
                logger.info(row)
            
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            avg_edge_rmse = np.mean(edge_rmse_values)
            avg_epi = np.mean(epi_values)
            
            logger.info("-" * 60)
            logger.info(f"{'Average':<10} {avg_psnr:<10.2f} {avg_ssim:<10.2f} {avg_edge_rmse:<15.4f} {avg_epi:<10.4f}")

        except Exception as e:
            logger.error(f"Error during visualization: {e}")

    def save_model(self, name: str) -> None:
        """Saves the generator and discriminator models."""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            self.generator.save(f'{MODEL_DIR}/{name}_generator.h5')
            self.discriminator.save(f'{MODEL_DIR}/{name}_discriminator.h5')
            logger.info(f"Models saved to '{MODEL_DIR}'.")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

def plot_loss_curves(history: dict, epochs: int, fold: int) -> None:
    """Plots and saves the training and validation loss curves."""
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Plot Generator Loss
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, epochs+1), history['train']['g_loss'], label='Train Generator Loss')
        plt.plot(range(1, epochs+1), history['val']['g_loss'], label='Validation Generator Loss')
        plt.title(f'Generator Loss During Training (Fold {fold})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/generator_loss_curve_fold_{fold}.png')
        plt.close()
        
        # Plot Discriminator Loss
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, epochs+1), history['train']['d_loss'], label='Train Discriminator Loss')
        plt.plot(range(1, epochs+1), history['val']['d_loss'], label='Validation Discriminator Loss')
        plt.title(f'Discriminator Loss During Training (Fold {fold})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/discriminator_loss_curve_fold_{fold}.png')
        plt.close()
        
        # Plot MAE
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, epochs+1), history['train']['mae'], label='Train MAE')
        plt.plot(range(1, epochs+1), history['val']['mae'], label='Validation MAE')
        plt.title(f'Mean Absolute Error During Training (Fold {fold})')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/mae_curve_fold_{fold}.png')
        plt.close()
        
        logger.info(f"Saved loss curves for fold {fold} to results directory.")
    except Exception as e:
        logger.error(f"Error plotting loss curves for fold {fold}: {e}")

def evaluate_final_model(model, test_noisy, test_clean, fold: int) -> Dict[str, Any]:
    """Calculate metrics on full test set after training"""
    # Generate all denoised images (single batch if possible)
    denoised = model.generator.predict(test_noisy, batch_size=32)  # Batch for efficiency
    
    # Vectorized metric calculations
    psnr_values = [psnr(c.squeeze(), d.squeeze(), data_range=1.0) 
                  for c, d in zip(test_clean, denoised)]
    
    ssim_values = [ssim(c.squeeze(), d.squeeze(), data_range=1.0)
                  for c, d in zip(test_clean, denoised)]
    
    # GPU-optimized edge metrics (if using TF)
    real_edges = model.sobel_edge_layer(test_clean).numpy()
    denoised_edges = model.sobel_edge_layer(denoised).numpy()
    edge_rmse_values = np.sqrt(np.mean((real_edges - denoised_edges)**2, axis=(1,2,3)))
    
    # EPI calculation (optimized)
    epi_values = [calculate_epi(c, d) for c, d in zip(test_clean, denoised)]
    
    # Aggregate statistics
    metrics = {
        'PSNR': {'mean': np.mean(psnr_values), 'std': np.std(psnr_values)},
        'SSIM': {'mean': np.mean(ssim_values), 'std': np.std(ssim_values)},
        'Edge_RMSE': {'mean': np.mean(edge_rmse_values), 'std': np.std(edge_rmse_values)},
        'EPI': {'mean': np.mean(epi_values), 'std': np.std(epi_values)},
        'num_samples': len(test_clean),
        'fold': fold
    }
    return metrics

def train(denoiser: MedicalDenoiser, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, 
          test_data: Tuple[np.ndarray, np.ndarray], epochs: int = EPOCHS, fold: int = 0) -> dict:
    """Training loop for the Medical Denoiser GAN."""
    os.makedirs(denoiser.checkpoint_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, f"summaries_fold_{fold}"))
    best_ssim = 0.0
    test_noisy, test_clean = test_data
    
    # Initialize dictionaries to store loss history
    history = {
        'train': defaultdict(list),
        'val': defaultdict(list)
    }

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs} (Fold {fold})")
        epoch_losses = {'d_loss': [], 'g_loss': [], 'mae': []}

        for batch, (noisy, clean) in enumerate(train_dataset):
            try:
                losses = denoiser.train_step((noisy, clean))
                for key, value in losses.items():
                    if np.isnan(value.numpy()):
                        raise ValueError(f"Loss '{key}' became NaN at batch {batch}.")
                for k, v in losses.items():
                    epoch_losses[k].append(float(v.numpy()))
                if batch % 50 == 0:
                    logger.info(f"Batch {batch}: D Loss: {losses['d_loss']:.4f}, G Loss: {losses['g_loss']:.4f}")
            except Exception as e:
                logger.error(f"Error during training batch {batch}: {e}")

        # Store training losses
        for k, v in epoch_losses.items():
            history['train'][k].append(np.mean(v))

        val_losses = {'d_loss': [], 'g_loss': [], 'mae': []}
        for noisy, clean in val_dataset:
            try:
                losses = denoiser.validation_step((noisy, clean))
                for k, v in losses.items():
                    val_losses[k].append(float(v.numpy()))
            except Exception as e:
                logger.error(f"Error during validation batch: {e}")

        # Store validation losses
        for k, v in val_losses.items():
            history['val'][k].append(np.mean(v))

        denoiser.visualize_results(test_noisy, test_clean, epoch)
        generated = denoiser.generator.predict(test_noisy[:5])
        current_ssim = np.mean([ssim(c.squeeze(), d.squeeze(), data_range=1.0) for c, d in zip(test_clean[:5], generated)])

        if current_ssim > best_ssim:
            denoiser.save_model(f'best_model_fold_{fold}')
            best_ssim = current_ssim

        denoiser.checkpoint.save(file_prefix=denoiser.checkpoint_prefix)

        with summary_writer.as_default():
            tf.summary.scalar('train/d_loss', np.mean(epoch_losses['d_loss']), step=epoch)
            tf.summary.scalar('train/g_loss', np.mean(epoch_losses['g_loss']), step=epoch)
            tf.summary.scalar('train/mae', np.mean(epoch_losses['mae']), step=epoch)
            tf.summary.scalar('val/d_loss', np.mean(val_losses['d_loss']), step=epoch)
            tf.summary.scalar('val/g_loss', np.mean(val_losses['g_loss']), step=epoch)
            tf.summary.scalar('val/mae', np.mean(val_losses['mae']), step=epoch)
            tf.summary.scalar('val/best_ssim', best_ssim, step=epoch)

        logger.info(f"Epoch {epoch + 1}/{epochs} (Fold {fold})")
        logger.info(f"Train - D Loss: {np.mean(epoch_losses['d_loss']):.4f} | "
                    f"G Loss: {np.mean(epoch_losses['g_loss']):.4f} | "
                    f"MAE: {np.mean(epoch_losses['mae']):.4f}")
        logger.info(f"Val - D Loss: {np.mean(val_losses['d_loss']):.4f} | "
                    f"G Loss: {np.mean(val_losses['g_loss']):.4f} | "
                    f"MAE: {np.mean(val_losses['mae']):.4f} | "
                    f"Best SSIM: {best_ssim:.4f}")
    
    # After training completes, plot and save the loss curves
    plot_loss_curves(history, epochs, fold)
    
    return history

def run_kfold_cross_validation(clean_images: np.ndarray, epochs: int = EPOCHS, k_folds: int = KFOLDS) -> Dict[str, Any]:
    """Run k-fold cross validation on the dataset."""
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    all_metrics = []
    
    for fold, (train_indices, test_indices) in enumerate(kfold.split(clean_images)):
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Fold {fold + 1}/{k_folds}")
        logger.info(f"{'='*50}")
        
        # Split data for this fold
        X_train_val = clean_images[train_indices]
        X_test = clean_images[test_indices]
        
        # Further split training data into train and validation sets
        X_train, X_val = train_test_split(X_train_val, test_size=VAL_SIZE, random_state=42)
        
        # Create noisy versions
        X_train_noisy = create_noisy_dataset(X_train)
        X_val_noisy = create_noisy_dataset(X_val)
        X_test_noisy = create_noisy_dataset(X_test)
        
        # Create datasets
        train_dataset = prepare_datasets(X_train, X_train_noisy)
        val_dataset = prepare_datasets(X_val, X_val_noisy)
        test_data = (X_test_noisy, X_test)
        
        # Initialize fresh model for each fold
        denoiser = MedicalDenoiser()
        
        # Train the model
        history = train(denoiser, train_dataset, val_dataset, test_data, epochs, fold)
        
        # Evaluate on test set
        metrics = evaluate_final_model(denoiser, X_test_noisy, X_test, fold)
        all_metrics.append(metrics)
        
        # Save fold results
        fold_results.append({
            'history': history,
            'metrics': metrics,
            'test_indices': test_indices
        })
        
        logger.info(f"\nFold {fold + 1} completed. Test metrics:")
        logger.info(f"PSNR: {metrics['PSNR']['mean']:.4f} ± {metrics['PSNR']['std']:.4f}")
        logger.info(f"SSIM: {metrics['SSIM']['mean']:.4f} ± {metrics['SSIM']['std']:.4f}")
        logger.info(f"Edge RMSE: {metrics['Edge_RMSE']['mean']:.4f} ± {metrics['Edge_RMSE']['std']:.4f}")
        logger.info(f"EPI: {metrics['EPI']['mean']:.4f} ± {metrics['EPI']['std']:.4f}")
    
    # Calculate overall cross-validation metrics
    overall_metrics = {
        'PSNR': {
            'mean': np.mean([m['PSNR']['mean'] for m in all_metrics]),
            'std': np.mean([m['PSNR']['std'] for m in all_metrics]),
            'all_folds': [m['PSNR']['mean'] for m in all_metrics]
        },
        'SSIM': {
            'mean': np.mean([m['SSIM']['mean'] for m in all_metrics]),
            'std': np.mean([m['SSIM']['std'] for m in all_metrics]),
            'all_folds': [m['SSIM']['mean'] for m in all_metrics]
        },
        'Edge_RMSE': {
            'mean': np.mean([m['Edge_RMSE']['mean'] for m in all_metrics]),
            'std': np.mean([m['Edge_RMSE']['std'] for m in all_metrics]),
            'all_folds': [m['Edge_RMSE']['mean'] for m in all_metrics]
        },
        'EPI': {
            'mean': np.mean([m['EPI']['mean'] for m in all_metrics]),
            'std': np.mean([m['EPI']['std'] for m in all_metrics]),
            'all_folds': [m['EPI']['mean'] for m in all_metrics]
        }
    }
    
    # Print overall results
    logger.info("\n\n" + "="*50)
    logger.info("Cross-Validation Results Summary")
    logger.info("="*50)
    logger.info(f"Average PSNR across all folds: {overall_metrics['PSNR']['mean']:.4f} ± {overall_metrics['PSNR']['std']:.4f}")
    logger.info(f"Average SSIM across all folds: {overall_metrics['SSIM']['mean']:.4f} ± {overall_metrics['SSIM']['std']:.4f}")
    logger.info(f"Average Edge RMSE across all folds: {overall_metrics['Edge_RMSE']['mean']:.4f} ± {overall_metrics['Edge_RMSE']['std']:.4f}")
    logger.info(f"Average EPI across all folds: {overall_metrics['EPI']['mean']:.4f} ± {overall_metrics['EPI']['std']:.4f}")
    
    # Plot cross-validation results
    plot_cross_validation_results(all_metrics)
    
    return {
        'fold_results': fold_results,
        'overall_metrics': overall_metrics
    }

def plot_cross_validation_results(all_metrics: List[Dict[str, Any]]) -> None:
    """Plot the cross-validation results across all folds."""
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Extract metrics for plotting
        folds = [f"Fold {m['fold']+1}" for m in all_metrics]
        psnr_means = [m['PSNR']['mean'] for m in all_metrics]
        ssim_means = [m['SSIM']['mean'] for m in all_metrics]
        edge_rmse_means = [m['Edge_RMSE']['mean'] for m in all_metrics]
        epi_means = [m['EPI']['mean'] for m in all_metrics]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PSNR plot
        axes[0, 0].bar(folds, psnr_means)
        axes[0, 0].set_title('PSNR Across Folds')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].grid(True)
        
        # SSIM plot
        axes[0, 1].bar(folds, ssim_means)
        axes[0, 1].set_title('SSIM Across Folds')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].grid(True)
        
        # Edge RMSE plot
        axes[1, 0].bar(folds, edge_rmse_means)
        axes[1, 0].set_title('Edge RMSE Across Folds')
        axes[1, 0].set_ylabel('Edge RMSE')
        axes[1, 0].grid(True)
        
        # EPI plot
        axes[1, 1].bar(folds, epi_means)
        axes[1, 1].set_title('EPI Across Folds')
        axes[1, 1].set_ylabel('EPI')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/cross_validation_results.png')
        plt.close()
        
        logger.info("Saved cross-validation results plot.")
    except Exception as e:
        logger.error(f"Error plotting cross-validation results: {e}")

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    logger.info("Loading and preprocessing medical images...")
    try:
        clean_images = load_medical_images(DATA_PATHS)
        
        # Run k-fold cross-validation
        cv_results = run_kfold_cross_validation(clean_images, epochs=EPOCHS, k_folds=KFOLDS)
        
        # Optionally, you can train a final model on all data after cross-validation
        logger.info("\nTraining final model on entire dataset...")
        X_train, X_test = train_test_split(clean_images, test_size=TEST_SIZE, random_state=42)
        X_train_noisy = create_noisy_dataset(X_train)
        X_test_noisy = create_noisy_dataset(X_test)
        
        train_dataset = prepare_datasets(X_train, X_train_noisy)
        test_data = (X_test_noisy, X_test)
        
        # For final training, we can use a validation split from the training data
        X_train, X_val = train_test_split(X_train, test_size=VAL_SIZE, random_state=42)
        X_train_noisy = create_noisy_dataset(X_train)
        X_val_noisy = create_noisy_dataset(X_val)
        
        train_dataset = prepare_datasets(X_train, X_train_noisy)
        val_dataset = prepare_datasets(X_val, X_val_noisy)
        
        final_model = MedicalDenoiser()
        final_history = train(final_model, train_dataset, val_dataset, test_data, epochs=EPOCHS, fold=-1)
        
        final_model.save_model('final_model')
        logger.info("Final model saved to models/final_model_generator.h5")
        
        # Evaluate final model
        final_metrics = evaluate_final_model(final_model, X_test_noisy, X_test, fold=-1)
        logger.info("\nFinal Model Test Metrics:")
        logger.info(f"PSNR: {final_metrics['PSNR']['mean']:.4f} ± {final_metrics['PSNR']['std']:.4f}")
        logger.info(f"SSIM: {final_metrics['SSIM']['mean']:.4f} ± {final_metrics['SSIM']['std']:.4f}")
        logger.info(f"Edge RMSE: {final_metrics['Edge_RMSE']['mean']:.4f} ± {final_metrics['Edge_RMSE']['std']:.4f}")
        logger.info(f"EPI: {final_metrics['EPI']['mean']:.4f} ± {final_metrics['EPI']['std']:.4f}")
        
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}")
```
