# X-GAN with Edge Preservation 
>**A Brief discussion of Novelty of the work and its medical adaptability**

## 1. Core Architecture Overview

The system implements a Generative Adversarial Network (GAN) with:

- A U-Net style generator with skip connections for detailed image reconstruction  
- A spectrally normalized discriminator for stable training  
- Edge attention mechanisms to preserve critical anatomical structures  
- Custom loss functions combining pixel, structural, and edge information

**Justification:**  
Medical images require precise preservation of edges and textures for accurate diagnosis. The hybrid architecture combines the strengths of GANs for realistic image generation with explicit edge preservation techniques.

---

## 2. Custom Functions and Components

### 2.1 Data Loading and Preprocessing

#### `normalize_medical_images(images)`
```python
def normalize_medical_images(images):
    p_min = np.percentile(images, 0.1)
    p_max = np.percentile(images, 99.9)
    normalized = np.clip(images, p_min, p_max)
    normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized) + 1e-7)
    return normalized
```
**Purpose:** Normalizes medical images while handling outliers.

**Justification:**
- Uses percentile clipping (0.1-99.9%) to exclude extreme values common in medical scans  
- Prevents contrast stretching from being affected by artifacts or noise  
- Maintains relative intensity relationships crucial for medical interpretation  

---

#### `add_xray_noise(images, peak)`
```python
def add_xray_noise(images, peak):
    scaled = images * peak  # Photon count scaling
    noisy = np.random.poisson(scaled).astype(np.float32) / peak
    return noisy
```
**Purpose:** Simulates realistic X-ray quantum noise.

**Justification:**
- Models the Poisson noise process inherent in X-ray imaging  
- Parameter `peak` controls noise level (higher = less noise)  
- Maintains physical accuracy of the noise model for realistic training  

---

#### `load_medical_images(data_paths)`
**Key Features:**
- Handles DICOM, PNG, JPEG formats  
- Automatic resizing and grayscale conversion  
- Robust error handling for corrupt files  

**Justification:**
- Medical datasets often contain mixed formats and corrupt files  
- Standardizes input size while preserving aspect ratio  
- Logs detailed loading statistics for quality control  

---

### 2.2 Custom Layers

#### `SobelEdgeLayer`
```python
class SobelEdgeLayer(layers.Layer):
    def call(self, inputs):
        sobel = tf.image.sobel_edges(inputs)
        return tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
```
**Purpose:** Computes edge magnitude maps using Sobel operators.

**Justification:**
- Provides differentiable edge detection for loss calculations  
- Preserves edge orientation information  
- More efficient than separate horizontal/vertical convolutions  

---

#### `EdgeAttention`
```python
class EdgeAttention(layers.Layer):
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.sobel_edge_layer = SobelEdgeLayer()
        self.conv = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid')

    def call(self, x):
        sobel_mag = self.sobel_edge_layer(x)
        att = self.conv(tf.expand_dims(sobel_mag, -1))
        return x * tf.tile(att, [1, 1, 1, self.input_channels])
```
**Purpose:** Applies attention to edge regions during processing.

**Justification:**
- Dynamically emphasizes edge features during denoising  
- Learned attention adapts to different edge types (soft/hard)  
- Helps preserve diagnostically critical boundaries  

---

#### `SpectralNormalization`
```python
class SpectralNormalization(tf.keras.layers.Wrapper):
    def build(self, input_shape):
        self.w = self.layer.kernel
        self.u = self.add_weight(shape=(1, self.w.shape[-1]), initializer='random_normal', trainable=False)

    def call(self, inputs, training=None):
        w_reshaped = tf.reshape(self.w, [-1, self.w.shape[-1]])
        for _ in range(1):  # One iteration of power method
            v = tf.nn.l2_normalize(tf.matmul(self.u, tf.transpose(w_reshaped)))
            self.u.assign(tf.nn.l2_normalize(tf.matmul(v, w_reshaped)))
        sigma = tf.matmul(tf.matmul(self.u, w_reshaped), tf.transpose(v))
        self.layer.kernel = self.w / sigma
        return self.layer(inputs)
```
**Purpose:** Stabilizes GAN training via weight normalization.

**Justification:**
- Enforces Lipschitz constraint on discriminator  
- Prevents mode collapse common in GANs  
- More stable than gradient penalty alone  

---

### 2.3 Loss Functions

#### `compute_loss(real, generated, real_edges, gen_edges)`
```python
def compute_loss(real, generated, real_edges, gen_edges):
    mae = tf.reduce_mean(tf.abs(real - generated))  # Pixel-level
    edge_loss = tf.reduce_mean(tf.abs(real_edges - gen_edges))  # Edge-level 
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(real, generated, max_val=1.0))  # Structural
    return 1.0 * mae + 5.0 * edge_loss + 2.0 * ssim_loss
```
**Components:**
- MAE (L1): Preserves pixel-level accuracy  
- Edge Loss: Explicitly maintains edge structures  
- SSIM: Preserves perceptual image quality  

**Justification:**
- Hybrid loss addresses multiple medical imaging needs  
- Edge loss weight (5.0) emphasizes structure preservation  
- SSIM improves perceptual quality over MSE alone  

---

### 2.4 Evaluation Metrics

#### `calculate_epi(original, denoised)`
```python
def calculate_epi(original, denoised):
    grad_orig = np.gradient(original)
    grad_denoised = np.gradient(denoised)
    scores = []
    for go, gd in zip(grad_orig, grad_denoised):
        numerator = np.sum(go * gd)
        denominator = np.sqrt(np.sum(go**2)) * np.sqrt(np.sum(gd**2))
        scores.append(numerator / (denominator + 1e-7))
    return np.mean(scores)
```
**Purpose:** Quantifies edge preservation quality.

**Justification:**
- More clinically relevant than pixel-wise metrics  
- Measures both edge strength and position accuracy  
- Windowed approach handles local variations  

---

#### `evaluate_final_model()`
**Metrics Reported:**
- PSNR - Pixel-level accuracy  
- SSIM - Structural similarity  
- Edge RMSE - Edge preservation  
- EPI - Clinical edge quality  

**Justification:**
- Comprehensive evaluation from pixels to clinical relevance  
- Provides both quantitative metrics and visual results  
- Standardized reporting enables comparison  

---

### 2.5 Training Framework

#### `train_step()`
**Key Features:**
- Gradient penalty for WGAN-GP stability  
- Separate optimizers for generator/discriminator  
- Gradient clipping prevents explosions  

**Justification:**
- WGAN-GP provides more stable training than standard GAN  
- Allows different learning rates for G/D  
- Careful gradient management for medical applications  

---

#### `run_kfold_cross_validation()`
**Implementation:**
- Stratified k-fold splitting  
- Fresh model per fold  
- Aggregated metrics  
- Final model on full data  

**Justification:**
- Robust performance estimation  
- Avoids overfitting to single split  
- Final model utilizes all available data  

---

## 3. Key Design Choices and Tradeoffs

### 3.1 Architecture Decisions

- **U-Net Generator:**  
  - Pros: Preserves fine details via skip connections  
  - Cons: Higher memory requirements than simple CNN  

- **PatchGAN Discriminator:**  
  - Pros: Focuses on local image statistics  
  - Cons: Less global coherence control  

- **Edge Attention:**  
  - Pros: Explicit structure preservation  
  - Cons: Adds computational overhead  

---

### 3.2 Clinical Considerations

- **Noise Model:**  
  - Realistic Poisson noise better than Gaussian for X-ray  
  - But may need adjustment for other modalities  

- **Normalization:**  
  - Percentile clipping handles scanner variations  
  - Preserves relative contrast important for diagnosis  

- **Evaluation:**  
  - EPI correlates with radiologist assessments  
  - Combines with traditional metrics for completeness  
