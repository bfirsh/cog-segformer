from transformers import (
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    ImageSegmentationPipeline,
)
from PIL import Image
from cog import BasePredictor, Input, Path, BaseModel, File
import torch
import io
from torch import nn
from typing import Any, List


class Segment(BaseModel):
    score: Any
    label: str
    mask: File


class Predictor(BasePredictor):
    def setup(self):
        feature_extractor = SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        self.image_segmenter = ImageSegmentationPipeline(
            model=model, feature_extractor=feature_extractor
        )

    def predict(self, image: Path) -> List[Segment]:
        output = self.image_segmenter(Image.open(image))
        for segment in output:
            image = segment["mask"]
            b = io.BytesIO()
            b.name = "image.png"
            image.save(b, "png")
            b.seek(0)
            segment["mask"] = b
        return output

