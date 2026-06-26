"""
CLAP: Contrastive Language-Audio Pretraining.

Reference: Wu et al., 2023 (https://arxiv.org/abs/2211.06687)

The audio analogue of CLIP: encodes audio clips into a shared text-audio
embedding space using an HTSAT audio encoder trained contrastively on
~630k audio-text pairs (LAION-Audio-630k).

Because the space is shared with text, a CLAP engine can embed natural-language
queries (embed_text) AND audio (embed_audio) into the SAME space, enabling
text-to-audio search, not just audio-to-audio. Embeddings are L2-normalised so
cosine similarity equals dot product.
"""

import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor

from src.engines.base import BaseEngine

# CLAP was trained on 48kHz audio; the processor expects this sample rate.
CLAP_SAMPLE_RATE = 48000


class CLAPEngine(BaseEngine):
    """CLAP audio (and text) encoder using laion/clap-htsat-unfused.

    Parameters
    ---
    model_name : HuggingFace model id (default laion/clap-htsat-unfused)
    batch_size : Number of clips per forward pass (default 16)
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        batch_size: int = 16,
    ):
        print(f"Loading CLAP model: {model_name}...")
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        self.batch_size = batch_size
        self.vector_size = self.model.config.projection_dim
        print(f"  Ready, embedding dim: {self.vector_size}")

    def fit(self, corpus: list[str]) -> None:
        print(f"CLAPEngine ({self.model_name}) is ready (no fitting required).")

    @staticmethod
    def _safe_load(path: str) -> np.ndarray:
        """Load mono audio resampled to 48kHz, returning silence on failure."""
        try:
            y, _ = librosa.load(path, sr=CLAP_SAMPLE_RATE, mono=True)
            return y if y.size > 0 else np.zeros(CLAP_SAMPLE_RATE, dtype=np.float32)
        except Exception:
            return np.zeros(CLAP_SAMPLE_RATE, dtype=np.float32)

    def _normalise(self, features: "torch.Tensor") -> np.ndarray:
        norms = features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (features / norms).cpu().numpy()

    # Project backbone pooler output into the shared 512-dim space explicitly.
    # (transformers 5.x get_audio_features/get_text_features return the raw
    # backbone output, so we apply the projection heads ourselves.)
    def _audio_features(self, processed) -> np.ndarray:
        outputs = self.model.audio_model(
            input_features=processed["input_features"],
            is_longer=processed.get("is_longer"),
        )
        features = self.model.audio_projection(outputs.pooler_output)
        return self._normalise(features)

    def _text_features(self, text_inputs) -> np.ndarray:
        outputs = self.model.text_model(**text_inputs)
        features = self.model.text_projection(outputs.pooler_output)
        return self._normalise(features)

    def embed_audio(self, audio_path: str) -> np.ndarray:
        audio = self._safe_load(audio_path)
        processed = self.processor(
            audio=audio, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
        )
        with torch.no_grad():
            return self._audio_features(processed)[0]

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a natural-language query into the shared text-audio space."""
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True)
        with torch.no_grad():
            return self._text_features(text_inputs)[0]

    def embed_batch(self, inputs: list[str]) -> np.ndarray:
        results = []
        for i in range(0, len(inputs), self.batch_size):
            batch_paths = inputs[i : i + self.batch_size]
            audios = [self._safe_load(p) for p in batch_paths]
            processed = self.processor(
                audio=audios, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                results.append(self._audio_features(processed))
        return np.concatenate(results, axis=0)
