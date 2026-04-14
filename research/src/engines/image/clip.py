"""
CLIP — Contrastive Language-Image Pretraining.

Reference: Radford et al., 2021 (https://arxiv.org/abs/2103.00020)

Encodes images into a 512-dim embedding space using a ViT-B/32 transformer
trained contrastively on 400M image-text pairs.
Embeddings are L2-normalised so cosine similarity equals dot product.
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.engines.base import BaseEngine


class CLIPEngine(BaseEngine):
    """CLIP image encoder using openai/clip-vit-base-patch32.

    Parameters
    ----------
    model_name : str
        HuggingFace model id (default ViT-B/32, ~600MB).
    batch_size : int
        Number of images per forward pass (default 32).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 32,
    ):
        print(f"Loading CLIP model: {model_name}...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        self.batch_size = batch_size
        self.vector_size = self.model.config.projection_dim
        print(f"  Ready — embedding dim: {self.vector_size}")

    def fit(self, corpus: list[str]) -> None:
        print(f"CLIPEngine ({self.model_name}) is ready (no fitting required).")

    def _image_features(self, pixel_values: "torch.Tensor") -> "torch.Tensor":
        """Extract L2-normalised image features via vision model + projection."""
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        features = self.model.visual_projection(vision_outputs.pooler_output)
        norms = features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return features / norms

    @staticmethod
    def _safe_open(path: str) -> Image.Image:
        """Open an image, returning a blank RGB image on any failure."""
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224))

    def embed_image(self, image_path: str) -> np.ndarray:
        pixel_values = self.processor(images=self._safe_open(image_path), return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            features = self._image_features(pixel_values)
        return features[0].cpu().numpy()

    def embed_batch(self, inputs: list[str]) -> np.ndarray:
        results = []
        for i in range(0, len(inputs), self.batch_size):
            batch_paths = inputs[i : i + self.batch_size]
            images = [self._safe_open(p) for p in batch_paths]
            pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                features = self._image_features(pixel_values)
            results.append(features.cpu().numpy())
        return np.concatenate(results, axis=0)
