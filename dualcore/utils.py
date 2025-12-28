import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from typing import List, Optional, Union
import torch

class EmbeddingManager:
    """Manages embedding models and caching."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text string."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return self.normalize(embedding)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Batch get embeddings."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Normalize each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

_default_manager: Optional[EmbeddingManager] = None

def get_default_embedding_manager() -> EmbeddingManager:
    """Singleton for default embedding manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = EmbeddingManager()
    return _default_manager

def get_embedding(text: str) -> np.ndarray:
    """Convenience function to get normalized embedding."""
    return get_default_embedding_manager().get_embedding(text)
