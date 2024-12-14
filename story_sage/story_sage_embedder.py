from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import torch

class StorySageEmbedder(EmbeddingFunction):
    """Embedding function using SentenceTransformer for generating text embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the StorySageEmbedder with a specified model.

        Args:
            model_name (str): Name of the SentenceTransformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the input documents.

        Args:
            input (Documents): Documents to generate embeddings for.

        Returns:
            Embeddings: List of embeddings for the input documents.
        """
        return self.model.encode(input).tolist()