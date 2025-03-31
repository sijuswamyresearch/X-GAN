import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_epi(original: np.ndarray, denoised: np.ndarray, window_size: int = 5) -> float:
    """
    Computes the Edge Preservation Index (EPI) between original and denoised images.
    Note: This code take approximately 5 minutes to calculate EPI measures of a 256X256 images
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

