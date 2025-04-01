import argparse
from scripts.train import main as train_model
from scripts.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    else:
        evaluate_model()
