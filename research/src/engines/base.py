import numpy as np


class BaseEngine:
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, inputs: list[str]) -> np.ndarray:
        """Embed a list of inputs (text strings or file paths depending on modality).
        Default falls back to embed_text — override for efficiency or for
        image/audio engines that implement embed_image / embed_audio instead."""
        return np.array([self.embed_text(t) for t in inputs])

    def embed_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError("Image embedding not supported by this engine")

    def embed_audio(self, audio_path: str) -> np.ndarray:
        raise NotImplementedError("Audio embedding not supported by this engine")
