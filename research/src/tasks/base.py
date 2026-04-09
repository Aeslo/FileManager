from src.engines.base import BaseEngine

class BaseTask:
    def run(self, engine: BaseEngine, dataset: object) -> dict[str, float]:
        raise NotImplementedError
