import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data_loader import DataLoaderWrapper
from config import Config
from blip_model import BLIPModel

def train():
    # Load configuration
    config = Config.load()

    # Initialize data loaders
    data_loaders = DataLoaderWrapper(
        train_image_dir=config['train_image_dir'],
        train_annotations=config['train_annotations'],
        val_image_dir=config['val_image_dir'],
        val_annotations=config['val_annotations'],
        test_image_dir=config['test_image_dir'],
        test_annotations=config['test_annotations'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        mean=config['mean'],
        std=config['std']
    )
    train_loader = data_loaders.get_dataloader('train')
    val_loader = data_loaders.get_dataloader('val')

    # Initialize model, optimizer, and loss function
    model = BLIPModel()
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{config['epochs']}"):
            pixel_values = batch['pixel_values'].to(model.device)
            captions = batch['captions']

            optimizer.zero_grad()
            loss = model(pixel_values, captions)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping for stability
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pixel_values = batch['pixel_values'].to(model.device)
                captions = batch['captions']

                loss = model(pixel_values, captions)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_model(os.path.join(checkpoint_dir, f"blip_epoch_{epoch + 1}.pth"))

if __name__ == "__main__":
    train()