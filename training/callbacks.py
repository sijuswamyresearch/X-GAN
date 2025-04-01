import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from .layers import SobelEdgeLayer  # Assuming this is available

class VisualizeCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, results_dir, num_samples=5):
        """
        Args:
            test_data: Tuple of (noisy_images, clean_images)
            results_dir: Directory to save visualizations
            num_samples: Number of samples to visualize each epoch
        """
        self.test_noisy, self.test_clean = test_data  # Unpack properly
        self.results_dir = results_dir
        self.num_samples = min(num_samples, len(self.test_noisy))
        os.makedirs(results_dir, exist_ok=True)
        self.edge_layer = SobelEdgeLayer()  # For edge preservation metrics

    def on_epoch_end(self, epoch, logs=None):
        # Generate denoised images
        denoised = self.model.generator.predict(self.test_noisy[:self.num_samples])
        
        # Calculate metrics
        metrics = []
        for clean, noisy, denoised_img in zip(self.test_clean[:self.num_samples], 
                                            self.test_noisy[:self.num_samples],
                                            denoised):
            clean = clean.squeeze()
            noisy = noisy.squeeze()
            denoised_img = denoised_img.squeeze()
            
            metrics.append({
                'psnr': psnr(clean, denoised_img, data_range=1.0),
                'ssim': ssim(clean, denoised_img, data_range=1.0),
                'noise_psnr': psnr(clean, noisy, data_range=1.0),  # Compare noisy vs clean
                'edge_rmse': self._calculate_edge_rmse(clean, denoised_img)
            })

        # Visualization
        plt.figure(figsize=(20, 4*self.num_samples))
        for i in range(self.num_samples):
            # Original Clean
            plt.subplot(self.num_samples, 4, i*4 + 1)
            plt.imshow(self.test_clean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Clean Target\nPSNR: {metrics[i]['psnr']:.2f}\nSSIM: {metrics[i]['ssim']:.2f}")
            plt.axis('off')
            
            # Noisy Input
            plt.subplot(self.num_samples, 4, i*4 + 2)
            plt.imshow(self.test_noisy[i].squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Noisy Input\nPSNR: {metrics[i]['noise_psnr']:.2f}")
            plt.axis('off')
            
            # Denoised Output
            plt.subplot(self.num_samples, 4, i*4 + 3)
            plt.imshow(denoised[i].squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title("Denoised Output")
            plt.axis('off')
            
            # Edge Difference
            plt.subplot(self.num_samples, 4, i*4 + 4)
            clean_edges = self.edge_layer(self.test_clean[i][np.newaxis, ...]).numpy().squeeze()
            denoised_edges = self.edge_layer(denoised[i][np.newaxis, ...]).numpy().squeeze()
            edge_diff = np.abs(clean_edges - denoised_edges)
            plt.imshow(edge_diff, cmap='hot')
            plt.title(f"Edge Differences\nRMSE: {metrics[i]['edge_rmse']:.4f}")
            plt.axis('off')
            plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/epoch_{epoch+1}.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Print metrics
        print(f"\nEpoch {epoch+1} Sample Metrics:")
        print(f"{'Sample':<8} {'PSNR':<8} {'SSIM':<8} {'Noise PSNR':<12} {'Edge RMSE':<10}")
        for i, m in enumerate(metrics):
            print(f"{i+1:<8} {m['psnr']:<8.2f} {m['ssim']:<8.2f} {m['noise_psnr']:<12.2f} {m['edge_rmse']:<10.4f}")
        print("-"*60)

    def _calculate_edge_rmse(self, clean, denoised):
        clean_edges = self.edge_layer(clean[np.newaxis, ..., np.newaxis]).numpy().squeeze()
        denoised_edges = self.edge_layer(denoised[np.newaxis, ..., np.newaxis]).numpy().squeeze()
        return np.sqrt(np.mean((clean_edges - denoised_edges)**2))
