import os
import argparse
from src.data.data_loader import DataLoader
from src.models.blip_model import BLIPModel
from src.models.training import train
from src.models.evaluation import evaluate
from src.utils.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning with BLIP")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate"],
                        help="Choose the mode: train or evaluate")
    parser.add_argument("--config", type=str, default="src/utils/config.py",
                        help="Path to the configuration file")

    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config)

    # Data Loading
    print("Loading data...")
    data_loader = DataLoader(config)
    train_data, val_data, test_data = data_loader.load_data()

    # Initialize Model
    print("Initializing model...")
    model = BLIPModel(config)

    if args.mode == "train":
        print("Starting training...")
        train(model, train_data, val_data, config)
    elif args.mode == "evaluate":
        print("Evaluating model...")
        evaluate(model, test_data, config)