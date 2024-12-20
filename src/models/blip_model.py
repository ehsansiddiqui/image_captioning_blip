import torch
import torch.nn as nn
from transformers import BlipForConditionalGeneration, BlipProcessor

class BLIPModel(nn.Module):
    def __init__(self, model_name='Salesforce/blip-image-captioning-base', device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(BLIPModel, self).__init__()
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def forward(self, pixel_values, captions=None):
        if captions is not None:
            # Training mode
            inputs = self.processor(text=captions, images=pixel_values, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            return loss
        else:
            # Inference mode
            inputs = self.processor(images=pixel_values, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**inputs)
            generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return generated_captions

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    def load_model(self, load_path):
        self.model = BlipForConditionalGeneration.from_pretrained(load_path).to(self.device)
        self.processor = BlipProcessor.from_pretrained(load_path)
