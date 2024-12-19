# Import necessary libraries and modules
import logging
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import torch

class StorySageEmbedder(EmbeddingFunction):
    """Embedding function using SentenceTransformer for generating text embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', *args, **kwargs):
        """
        Initialize the StorySageEmbedder with a specified SentenceTransformer model.

        Args:
            model_name (str): Name of the SentenceTransformer model to use.
        """
        # Call the parent class initializer
        super().__init__(*args, **kwargs)
        # Set up logging for debugging purposes
        self.logger = logging.getLogger(__name__)
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        # Determine the device to run the model on (GPU if available, else CPU)
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        # Move the model to the selected device
        self.model = self.model.to(self.device)

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the input documents.

        Args:
            input (Documents): A list of documents (texts) to generate embeddings for.

        Returns:
            Embeddings: List of embeddings for the input documents.
        """
        # Log the number of texts to embed
        self.logger.debug(f"Embedding {len(input)} texts.")
        # Generate embeddings using the model
        embeddings = self.model.encode(input).tolist()
        # Log that embedding is completed
        self.logger.debug("Embedding completed.")
        # Return the embeddings
        return embeddings