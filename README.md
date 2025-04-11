![X-GAN](X-GAN-title.png)
![X-GAN](X-GAN-result.png)
## Project repository for X-GAN

This repository contains the complete code, results and models from the X-GAN project. This project is done by Mr. Siju K. S, Research Scholar, Amrita School of Artificial Intelligence and is supervised by [Dr. Vipin Venugopal](https://sites.google.com/view/vipin-venugopal?pli=1) , Assistant Professor (Sel.Gr), Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore.


## Project Documentation

# Medical Image Denoising Using X-GAN

## Project Overview

This project implements a **Generative Adversarial Network (GAN)** for medical image denoising. The model, named **X-GAN**, leverages advanced techniques such as **Spectral Normalization**, **Edge Preservation Index (EPI)**, and **Custom Sobel Edge Layers** to produce high-quality denoised images while preserving critical structural details. The architecture consists of:

- A **U-Net-like generator** for denoising.
- A **PatchGAN discriminator** for adversarial training.
- Custom loss functions combining **adversarial loss**, **reconstruction loss**, and **gradient penalty**.

The project is designed to handle realistic noise models in medical imaging datasets and provides tools for evaluation metrics such as **PSNR**, **SSIM**, and **Edge RMSE**.

---

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- Dependencies: `numpy`, `scikit-image`, `matplotlib`, `opencv-python`, `tqdm`

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/sijuswamyresearch/X-GAN.git
   cd medical_image_denoising
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   

## 📁 Repository Structure

```
medical_image_denoising/
│
├── configs/
│   └── default.yaml       # Hyperparameters and paths
│
├── data/
│   ├── __init__.py
│   ├── dataloader.py      # Data loading and preprocessing
│   └── augmentations.py   # Noise addition and transforms
│
├── models/
│   ├── __init__.py
│   ├── generator.py       # Generator architecture
│   ├── discriminator.py   # Discriminator architecture
│   ├── xgan.py           # Main GAN system
│   └── layers.py         # Custom layers
│
├── training/
│   ├── __init__.py
│   ├── trainer.py        # Training loop
│   ├── losses.py         # Loss functions
│   └── callbacks.py      # Custom callbacks
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py        # EPI, PSNR, SSIM calculations
│   ├── visualize.py      # Plotting functions
│   └── logger.py         # Logging setup
│
├── scripts/
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
│
└── main.py               # Entry point
```

