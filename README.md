# X-GAN

## The Project repository for X-GAN

This repository contains the complete code, results and models from the X-GAN project. This project is done by Mr. Siju K. S, Research Scholar, Amrita School of Artificial Intelligence and is supervised by [Dr. Vipin Venugopal](https://sites.google.com/view/vipin-venugopal?pli=1) , Assistant Professor (Sel.Gr), Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore.

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
