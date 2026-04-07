import numpy as np

class BaseEngine:
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError
    
    def embed_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError("Image embedding not supported by this engine")
    
    def embed_audio(self, audio_path: str) -> np.ndarray:
        raise NotImplementedError("Audio embedding not supported by this engine")
