from models.xgan import XGAN
from data.dataloader import load_dataset
from training.trainer import XGANTrainer
from configs import load_config

def main():
    config = load_config()
    
    # Load data
    train_ds, val_ds, test_ds = load_dataset(config)
    
    # Initialize model
    model = XGAN(config)
    
    # Train
    trainer = XGANTrainer(model, train_ds, val_ds, config)
    history = trainer.train(config['epochs'])
    
    # Save and evaluate
    model.save()
    model.evaluate(test_ds)

if __name__ == "__main__":
    main()
