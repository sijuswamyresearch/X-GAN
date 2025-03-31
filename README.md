# X-GAN

## The Project repository for X-GAN

This repository contains the complete code, results and models from the X-GAN project. This project is done by Mr. Siju K. S, Research Scholar, Amrita School of Artificial Intelligence and is supervised by [Dr. Vipin Venugopal](https://sites.google.com/view/vipin-venugopal?pli=1) , Assistant Professor (Sel.Gr), Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore.

## 📁 Repository Structure

```
xgan_denoising/
│
├── models/
│   ├── generator.py       # Generator network
│   ├── discriminator.py   # Discriminator network
│   └── xgan.py           # Main GAN system
│
├── training/
│   ├── train_utils.py     # Losses, metrics
│   ├── callbacks.py       # Custom callbacks
│   └── trainer.py        # Training loop
│
├── data/
│   ├── dataloader.py      # Data pipeline
│   └── preprocessing.py   # Noise addition, etc.
│
├── configs/
│   └── default.yaml       # Hyperparameters
│
└── utils/
    ├── visualize.py       # Result plotting
    └── metrics.py        # EPI, PSNR, SSIM
```
