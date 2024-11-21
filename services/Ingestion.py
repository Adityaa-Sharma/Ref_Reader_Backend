import nltk
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
import requests
import PyPDF2
import io
from typing import Dict, List, Any, Generator

class VectorIngestor:
    def __init__(self, session_id: str, paper_name: str, arxiv_id: str):
        """Initialize the Qdrant client and OpenAI embeddings for vector ingestion."""
        try:
            self.session_id = session_id
            self.client = QdrantClient(host='localhost', port=6333)
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"  # Changed from model_name to model
            )
            self.paper_name = paper_name
            self.arxiv_id = arxiv_id
            # Create collection synchronously in __init__
            self._create_collection()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize VectorIngestor: {str(e)}")
    
    def _create_collection(self):
        """Synchronous version of create collection."""
        try:
            try:
                collection_info = self.client.get_collection(self.session_id)
                if collection_info:
                    print(f"Collection '{self.session_id}' already exists.")
                    return
            except Exception as e:
                if "Not found" in str(e):
                    self.client.recreate_collection(
                        collection_name=self.session_id,
                        vectors_config={"size": 1536, "distance": "Cosine"},
                    )
                    print(f"Collection '{self.session_id}' created successfully.")
                else:
                    raise e
        except Exception as e:
            print(f"Error in create_collection: {str(e)}")
            raise

    async def arxiv_handling(self, arxiv_id: str) -> Dict[str, Any]:
        """Process arxiv paper and ingest it into vector store."""
        try:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch PDF from arXiv: {response.status_code}"
                )
            
            pdf_content = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_content)
            
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + " "
            
            result = await self.process_and_ingest_text(full_text)
            return result
            
        except Exception as e:
            error_message = f"Error processing PDF from arXiv: {str(e)}"
            raise HTTPException(status_code=500, detail=error_message)

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings synchronously."""
        text=f"{self.paper_name}\n{text}"
        return self.embeddings.embed_query(text)

    def generate_points(self, chunks: List[str], batch_size: int = 100) -> Generator[List[PointStruct], None, None]:
        """Generate points synchronously."""
        points = []
        for chunk in chunks:
            ## attach the paper name to the chunk with a line break
            # chunk = f"{{\"paper_name\": self.paper_name}} {chunk}"
            # chunk = f"{self.paper_name}\n{chunk}"
            
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

    async def process_and_ingest_text(self, text: str, batch_size: int = 1000) -> Dict[str, str]:
        """Process and ingest text into vector store."""
        try:
            text = ' '.join(text.split())
            words = nltk.word_tokenize(text)
            chunks = [' '.join(words[i:i + 500]) for i in range(0, len(words), 500)]
            
            for batch in self.generate_points(chunks, batch_size):
                self.client.upsert(collection_name=self.session_id, points=batch)

            return {
                "status": "success",
                "message": f"Successfully processed and ingested {len(chunks)} chunks for paper '{self.paper_name}' (arXiv ID: {self.arxiv_id})"
            }

        except Exception as e:
            error_message = f"Error during text processing or ingestion: {str(e)}"
            raise HTTPException(status_code=500, detail=error_message)