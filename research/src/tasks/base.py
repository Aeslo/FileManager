from typing import Any, Dict
from src.engines.base import BaseEngine

class BaseTask:
    def run(self, engine: BaseEngine, dataset: Any) -> Dict[str, float]:
        raise NotImplementedError
