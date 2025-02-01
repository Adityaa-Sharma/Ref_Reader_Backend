import nltk
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
import requests
import PyPDF2
import io
from typing import Dict, List, Any, Generator
import os
from dotenv import load_dotenv
import logging

# Configure logger
logger = logging.getLogger(__name__)

class VectorIngestor:
    def __init__(self, paper_name: str,main_arxiv_id: str, arxiv_id: str):
        load_dotenv()

        try:
            # Read configuration from environment variables
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
            embedding_deployment = os.getenv('AZURE_EMBEDDING_DEPLOYMENT', 'TG-OAI-Embedding')
            embedding_model = os.getenv('AZURE_EMBEDDING_MODEL', 'text-embedding-ada-002')
            qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
            qdrant_port = int(os.getenv('QDRANT_PORT', 6333))

            # Validate configuration
            if not all([azure_endpoint, azure_api_key,embedding_deployment, embedding_model, qdrant_host, qdrant_port]):
                raise ValueError("Missing Azure OpenAI configuration")

            # Initialize clients
            self.client = QdrantClient(host='localhost', port=6333)
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                deployment=embedding_deployment,
                model=embedding_model
            )
                
            self.paper_name = paper_name
            self.arxiv_id = arxiv_id
            self.main_arxiv_id = main_arxiv_id
            self._create_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorIngestor: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize VectorIngestor: {str(e)}"
            )
    
    def _create_collection(self):
        """Synchronous version of create collection."""
        try:
            collections = self.client.get_collections().collections
            exists = any(collection.name == self.arxiv_id for collection in collections)
            
            if not exists:
                logger.info(f"Creating collection {self.main_arxiv_id}")
                self.client.recreate_collection(
                    collection_name=self.main_arxiv_id,
                    vectors_config={
                        "size": 1536,
                        "distance": "Cosine"
                    }
                )
                
                # Verify collection was created
                collections = self.client.get_collections().collections
                if not any(collection.name == self.main_arxiv_id for collection in collections):
                    raise Exception(f"Failed to create collection {self.arxiv_id}")
                logger.info(f"Successfully created collection {self.arxiv_id}")
            else:
                logger.info(f"Collection {self.arxiv_id} already exists")

        except Exception as e:
            logger.error(f"Error in create_collection: {str(e)}")
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
            
            # Ensure collection exists before processing
            self._create_collection()
            
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
            # Ensure collection exists before processing
            self._create_collection()
            
            text = ' '.join(text.split())
            words = nltk.word_tokenize(text)
            chunks = [' '.join(words[i:i + 500]) for i in range(0, len(words), 500)]
            
            total_chunks = len(chunks)
            processed_chunks = 0
            
            for batch in self.generate_points(chunks, batch_size):
                self.client.upsert(collection_name=self.main_arxiv_id, points=batch)
                processed_chunks += len(batch)
                logger.info(f"Processed {processed_chunks}/{total_chunks} chunks")

            return {
                "status": "success",
                "message": f"Successfully processed and ingested {total_chunks} chunks for paper '{self.paper_name}' (arXiv ID: {self.arxiv_id}, in the collection {self.main_arxiv_id})"
            }

        except Exception as e:
            error_message = f"Error during text processing or ingestion: {str(e)}"
            raise HTTPException(status_code=500, detail=error_message)