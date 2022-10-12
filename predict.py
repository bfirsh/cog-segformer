from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
from cog import BasePredictor, Input, Path
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

    def predict(self, image: Path) -> Any:
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
        labels = upsampled_predictions.flatten().tolist()

        class_names = {}

        for cls in set(labels):
            if cls not in class_names:
                class_names[cls] = self.model.config.id2label[cls]

        return {"labels": labels, "class_names": class_names}

