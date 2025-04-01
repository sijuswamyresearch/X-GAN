import numpy as np
from models import MedicalDenoiser
from utils.metrics import calculate_psnr, calculate_ssim, calculate_epi

def evaluate_model(model, test_noisy, test_clean):
    denoised = model.generator.predict(test_noisy)
    
    psnrs = [calculate_psnr(c, d) for c, d in zip(test_clean, denoised)]
    ssims = [calculate_ssim(c, d) for c, d in zip(test_clean, denoised)]
    epis = [calculate_epi(c, d) for c, d in zip(test_clean, denoised)]
    
    print(f"PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f}")
    print(f"SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"EPI: {np.mean(epis):.4f} ± {np.std(epis):.4f}")
