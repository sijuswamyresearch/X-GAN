import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_epi(original, denoised, window_size=5):
    """Your improved EPI implementation"""
    original = original.squeeze().astype(np.float32)
    denoised = denoised.squeeze().astype(np.float32)
    
    # Gradient calculations
    grad_x_orig = cv2.Sobel(original, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_orig = cv2.Sobel(original, cv2.CV_32F, 0, 1, ksize=3)
    grad_orig = np.sqrt(grad_x_orig**2 + grad_y_orig**2)
    
    # ... rest of your EPI calculation ...
    
    return np.mean(epi_values) if epi_values else 0.0
