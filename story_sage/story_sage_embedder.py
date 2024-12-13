
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import torch

class StorySageEmbedder(EmbeddingFunction):
  def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
    self.model = SentenceTransformer(model_name)
    self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    self.model = self.model.to(self.device)

  def __call__(self, input: Documents) -> Embeddings:
      return self.model.encode(input).tolist()