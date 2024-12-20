import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import DataLoaderWrapper
from config import Config
from blip_model import BLIPModel

def test():
    # Load configuration
    config = Config.load()

    # Initialize data loader for testing
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
    test_loader = data_loaders.get_dataloader('test')

    # Load trained model
    model = BLIPModel()
    checkpoint_path = os.path.join(config['checkpoint_dir'], 'blip_best_model.pth')  # Adjust the path to the desired checkpoint
    model.load_model(checkpoint_path)
    model.eval()

    all_captions = []
    all_generated_captions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch['pixel_values'].to(model.device)
            captions = batch['captions']  # Ground truth captions

            # Generate captions using the model
            generated_captions = model(pixel_values)

            all_captions.extend(captions)
            all_generated_captions.extend(generated_captions)

    # Calculate evaluation metrics (e.g., BLEU, METEOR, CIDEr, ROUGE-L)
    evaluate_captions(all_captions, all_generated_captions)

def evaluate_captions(ground_truths, predictions):
    from nltk.translate.bleu_score import corpus_bleu
    from rouge_score import rouge_scorer

    references = [[caption.split()] for caption in ground_truths]  # Tokenize ground truth captions
    hypotheses = [caption.split() for caption in predictions]     # Tokenize generated captions

    # Compute BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"BLEU Score: {bleu_score:.4f}")

    # Compute ROUGE-L score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(ground_truths, predictions)]
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores)
    print(f"Average ROUGE-L Score: {avg_rouge_l:.4f}")

    # Placeholder for CIDEr and METEOR evaluation (additional libraries needed)
    print("For CIDEr and METEOR, consider integrating libraries such as pycocoevalcap.")

if __name__ == "__main__":
    test()