import numpy as np
from src.tasks.sorting import SortingTask
from src.engines.text.tfidf import TfidfEngine

def test_sorting_task():
    # Synthetic data
    data = [
        "Dogs are loyal domestic pets", 
        "Cats are common household pets", 
        "Satellite is in geostationary orbit", 
        "Rocket is entering deep space"
    ]
    labels = [0, 0, 1, 1]  # 0 = Animals/Pets, 1 = Space
    
    engine = TfidfEngine()
    engine.fit(data)
    
    task = SortingTask(n_clusters=2)
    metrics = task.run(engine, (data, labels))
    
    assert "ari" in metrics
    # ARI should be high for this very simple example
    assert metrics["ari"] > 0.5
