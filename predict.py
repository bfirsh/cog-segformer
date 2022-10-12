from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
from cog import BasePredictor, Input, File
import torch
from torch import nn
from typing import Any


class Predictor(BasePredictor):
    def setup(self):
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )

    def predict(self, image: File) -> Any:
        image_obj = Image.open(image)
        inputs = self.feature_extractor(images=image_obj, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image_obj.size[::-1],
            mode="bilinear",
        )
        upsampled_predictions = upsampled_logits.argmax(dim=1) + 1
        return upsampled_predictions.flatten().tolist()
