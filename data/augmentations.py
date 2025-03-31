import numpy as np

def normalize_medical_images(images):
    p_min = np.percentile(images, 0.1)
    p_max = np.percentile(images, 99.9)
    normalized = np.clip(images, p_min, p_max)
    return (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized) + 1e-7)

def add_xray_noise(images, peak=1000):
    scaled = images * peak
    noisy = np.random.poisson(scaled).astype(np.float32) / peak
    return np.clip(noisy, 0, 1)
