import numpy as np
from typing import List

def normalize_medical_images(images: np.ndarray) -> np.ndarray:
    p_min = np.percentile(images, 0.1)
    p_max = np.percentile(images, 99.9)
    normalized = np.clip(images, p_min, p_max)
    normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized) + 1e-7)
    return normalized.astype(np.float32)

def add_xray_noise(images: np.ndarray, peak: int = 1000) -> np.ndarray:
    scaled = images * peak
    noisy = np.random.poisson(scaled).astype(np.float32) / peak
    return np.clip(noisy, 0, 1)
