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
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logger
logger = logging.getLogger(__name__)

class VectorIngestor:
    def __init__(self, paper_name: str,main_arxiv_id: str, arxiv_id: str):
        load_dotenv()

        try:
            # Azure OpenAI setup
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
            embedding_deployment = os.getenv('AZURE_EMBEDDING_DEPLOYMENT', 'TG-OAI-Embedding')
            embedding_model = os.getenv('AZURE_EMBEDDING_MODEL', 'text-embedding-ada-002')
            
            # Updated Qdrant connection logic for Docker
            logger.info("Initializing Qdrant connection...")
            
            try:
                # Use the service name from docker-compose
                self.client = QdrantClient(
                    host="qdrant",  # Changed from 127.0.0.1 to service name
                    port=6333,
                    timeout=10
                )
                # Test connection
                collections = self.client.get_collections()
                logger.info(f"Successfully connected to Qdrant. Collections: {collections}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {str(e)}")
                raise

            # Initialize embeddings and text splitter
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                deployment=embedding_deployment,
                model=embedding_model
            )
            
            self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            length_function=len,
            separators=[
                "\nAbstract",
                "\nIntroduction",
                "\nBackground",
                "\nMethodology",
                "\nResults",
                "\nDiscussion",
                "\nConclusion",
                "\nReferences",
                "\n\d+\.",      
                "\n\d+\.\d+", 
                "\nFigure \d+",
                "\nTable \d+",
                "\n\n",        
                "\n",          # Line breaks
                ". ",          # Sentences
                "; ",          # Semi-colons
                ": ",          # Colons
                ", ",          # Commas
                " ",           # Words
                ""            # Characters
            ],
            keep_separator=True
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

    def generate_embeddings(self, text: str, metadata: dict) -> List[float]:
        """Generate embeddings with metadata context."""
        context = f"Title: {metadata['paper_name']}\nArXiv ID: {metadata['arxiv_id']}\nContent: {text}"
        return self.embeddings.embed_query(context)

    def generate_points(self, chunks: List[dict], batch_size: int = 100) -> Generator[List[PointStruct], None, None]:
        """Generate points from chunks with metadata."""
        points = []
        for chunk in chunks:
            metadata = {
                "paper_name": self.paper_name,
                "arxiv_id": self.arxiv_id,
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"]
            }
            
            embedding = self.generate_embeddings(chunk["text"], metadata)
            point_id = str(uuid.uuid4())

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    **metadata
                }
            )
            points.append(point)

            if len(points) >= batch_size:
                yield points
                points = []

        if points:
            yield points

    async def process_and_ingest_text(self, text: str, batch_size: int = 100) -> Dict[str, str]:
        """Process and ingest text into vector store using smart chunking."""
        try:
            self._create_collection()
            
            # Clean and prepare text
            text = ' '.join(text.split())
            
            # Generate chunks using the text splitter
            chunks = self.text_splitter.split_text(text)
            total_chunks = len(chunks)
            
            # Prepare chunks with metadata
            processed_chunks = [
                {
                    "text": chunk,
                    "chunk_index": idx,
                    "total_chunks": total_chunks
                }
                for idx, chunk in enumerate(chunks)
            ]
            
            processed_count = 0
            for batch in self.generate_points(processed_chunks, batch_size):
                self.client.upsert(collection_name=self.main_arxiv_id, points=batch)
                processed_count += len(batch)
                logger.info(f"Processed {processed_count}/{total_chunks} chunks")

            return {
                "status": "success",
                "message": f"Successfully processed and ingested {total_chunks} chunks for paper '{self.paper_name}' (arXiv ID: {self.arxiv_id})",
                "details": {
                    "total_chunks": total_chunks,
                    "collection_name": self.main_arxiv_id,
                    "paper_name": self.paper_name,
                    "arxiv_id": self.arxiv_id
                }
            }

        except Exception as e:
            error_message = f"Error during text processing or ingestion: {str(e)}"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)