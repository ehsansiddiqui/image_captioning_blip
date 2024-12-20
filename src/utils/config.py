import os

class Config:
    @staticmethod
    def load():
        return {
            # Paths to datasets
            'train_image_dir': os.path.join('data', 'train', 'images'),
            'train_annotations': os.path.join('data', 'train', 'captions.json'),
            'val_image_dir': os.path.join('data', 'val', 'images'),
            'val_annotations': os.path.join('data', 'val', 'captions.json'),
            'test_image_dir': os.path.join('data', 'test', 'images'),
            'test_annotations': os.path.join('data', 'test', 'captions.json'),

            # Image preprocessing
            'image_size': 224,  # Resize images to 224x224
            'mean': [0.485, 0.456, 0.406],  # ImageNet mean values for normalization
            'std': [0.229, 0.224, 0.225],  # ImageNet std values for normalization

            # Training parameters
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 20,

            # Model settings
            'model_name': 'blip',  # Model identifier for future extension
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
        }