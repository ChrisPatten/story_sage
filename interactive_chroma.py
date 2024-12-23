import chromadb
from story_sage.story_sage_embedder import StorySageEmbedder
from langchain.embeddings import SentenceTransformerEmbeddings
class EmbeddingAdapter(SentenceTransformerEmbeddings):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _embed_documents(self, texts):
        return super().embed_documents(texts)  

    def __call__(self, input):
        return self._embed_documents(input)  


def get_vector_store():
  embedder = EmbeddingAdapter
  chroma_path = './chroma_data'
  chroma_collection = 'story_sage'
  client = chromadb.PersistentClient(path=chroma_path)
  vector_store = client.get_collection(name=chroma_collection)
  return vector_store

