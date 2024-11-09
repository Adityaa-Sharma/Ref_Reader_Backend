import nltk
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings

class VectorIngestor:
    def __init__(self, session_id: str, paper_name: str, arxiv_id: str):
        """Initialize the Qdrant client and OpenAI embeddings for vector ingestion."""
        try:
            self.session_id = session_id
            self.client = QdrantClient(host='localhost', port=6333)
            self.embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small")
            self.paper_name = paper_name
            self.arxiv_id = arxiv_id
            self.create_collection()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize VectorIngestor: {str(e)}")

    def create_collection(self):
        """Create a collection in Qdrant if it doesn't exist."""
        try:
            collection_info = self.client.get_collection(self.session_id)
            if collection_info:
                print(f"Collection '{self.session_id}' already exists.")
                return

        except Exception as e:
            if "Not found" in str(e):
                try:
                    # OpenAI text-embedding-3-small uses 1536 dimensions
                    self.client.recreate_collection(
                        collection_name=self.session_id,
                        vectors_config={"size": 1536, "distance": "Cosine"},
                    )
                    print(f"Collection '{self.session_id}' created successfully.")
                except Exception as e:
                    print(f"Failed to create collection '{self.session_id}': {str(e)}")
            else:
                print(f"Error checking for collection existence: {str(e)}")

    def generate_embeddings(self, text: str):
        """Generate embeddings using OpenAI's embedding model."""
        return self.embeddings.embed_query(text)

    def generate_points(self, chunks, batch_size=100):
        """Generate points from text chunks with embeddings for upsert."""
        points = []
        for chunk in chunks:
            embedding = self.generate_embeddings(chunk)
            point_id = str(uuid.uuid4())

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "paper_name": self.paper_name,
                    "arxiv_id": self.arxiv_id
                }
            )
            points.append(point)

            if len(points) >= batch_size:
                yield points
                points = []

        if points:
            yield points

    async def process_and_ingest_text(self, chunk: str, batch_size=1000):
        """
        Tokenize and chunk text, then upsert chunks into Qdrant with error handling.
        
        Args:
            chunk (str): Text content to process.
            batch_size (int): Number of points per batch for upsert.

        Returns:
            dict: Status and message indicating success or error details.
        """
        try:
  
            words = nltk.word_tokenize(chunk)
            
            chunks = [' '.join(words[i:i + 500]) for i in range(0, len(words), 500)]
            
            for batch in self.generate_points(chunks, batch_size):
                self.client.upsert(collection_name=self.session_id, points=batch)

            return {
                "status": "success",
                "message": f"Chunks for paper '{self.paper_name}' (arxiv ID: {self.arxiv_id}) ingested successfully."
            }

        except Exception as e:
            error_message = f"Error during text processing or ingestion: {str(e)}"
            raise HTTPException(status_code=500, detail=error_message)