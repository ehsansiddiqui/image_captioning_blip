# Image Captioning with BLIP (Bootstrapped Language-Image Pretraining)

## Project Overview
This project implements an image captioning system using the BLIP (Bootstrapped Language-Image Pretraining) model. The model generates descriptive captions for images by leveraging pre-trained multi-modal transformers. The project includes custom data handling, model training, and evaluation on datasets like COCO or any user-provided dataset.

## Dataset
This project supports multiple datasets for training, validation, and testing. By default, it is configured to use the COCO dataset, which contains images paired with captions describing their contents. The dataset is split as follows:

- **Training Set**: Used for training the model.
- **Validation Set**: Used for hyperparameter tuning and early stopping.
- **Test Set**: Used to evaluate the final performance of the trained model.

Each dataset consists of:
- Images stored in a directory.
- Annotations provided in a JSON file, which maps image filenames to their respective captions.

## Configuration
The configuration parameters are stored in a `config.py` file. These parameters include:
- File paths for datasets (images and annotations).
- Hyperparameters such as batch size, learning rate, and training epochs.
- Model-specific parameters like image size and normalization values.
- Directories for saving checkpoints and logs.

## Training Logic
The training pipeline is implemented in `train.py` and includes the following steps:

1. **Data Preparation**: Images are preprocessed using a transformer-compatible pipeline (e.g., resizing, normalization). Annotations are tokenized into text sequences.

2. **Model Initialization**: The BLIP model is initialized with pre-trained weights to leverage transfer learning. The decoder is fine-tuned to improve the captioning quality for the specific dataset.

3. **Training Loop**:
   - The model is trained using a cross-entropy loss function to minimize the difference between predicted and ground-truth captions.
   - Optimizers (e.g., AdamW) and learning rate schedulers are used to improve convergence.

4. **Validation**: After each epoch, the model is evaluated on the validation set to monitor performance and save the best-performing checkpoint.

## Evaluation Metrics
Evaluation is performed using the `test.py` script, which generates captions for the test images and compares them to ground-truth captions. The following metrics are used to assess the model's performance:

1. **BLEU (Bilingual Evaluation Understudy)**:
   - Measures the overlap of n-grams between generated and reference captions.
   - Computed using the `nltk` library.

2. **ROUGE-L**:
   - Evaluates the longest common subsequences between generated and reference captions.
   - Computed using the `rouge_score` library.

## How to Run the Project

### Prerequisites
- Python 3.8 or higher.
- PyTorch and related libraries.
- Datasets such as COCO or custom datasets prepared in the required format.

## Acknowledgments
- The BLIP model implementation is inspired by the original research and available resources.
- Dataset preparation and evaluation methods adhere to the standards of the COCO dataset.

Feel free to contribute by reporting issues or suggesting improvements!

