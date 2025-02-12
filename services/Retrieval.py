import os
import asyncio
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI,AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retrieval:
    def __init__(self, query: Union[str, Any], arxiv_id :str,arxiv_id_main:str, chat_history: str = ""):
        load_dotenv()  
        
        # Ensure query is a string
        self.query = str(query) if query is not None else ""
        
        # Update Qdrant client configuration
        try:
            # Use container name from docker-compose.yml instead of localhost
            qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')  # Changed from 'localhost' to 'qdrant'
            qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
            
            logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
            
            self.client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                timeout=10  # Add timeout
            )
            
            # Verify connection
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise RuntimeError(f"Database connection failed: {str(e)}")
        
        # OpenAI embeddings configuration
        # self.embeddings = OpenAIEmbeddings(
        #     model="text-embedding-3-small",
        #     api_key=os.getenv('OPENAI_API_KEY')
        # )
        self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                deployment="TG-OAI-Embedding",
                model="text-embedding-ada-002"
            )
        # Language model configuration
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("LLM_MODEL_NAME"),
            api_version="2024-08-01-preview",
            model_kwargs={
                "response_format": { "type": "text" }
            }
            
            
        )
        self.arxiv_id = arxiv_id
        self.arxiv_id_main = arxiv_id_main
        
        # Instance variables
        # self.session_id = session_id
        self.chat_history = chat_history

    async def retrieve(self) -> List[str]:
        """
        Asynchronously retrieve relevant documents from Qdrant collection.
        
        :return: List of retrieved document texts
        """
        try:
            # Verify collection exists
            collections = self.client.get_collections()
            if not any(col.name == self.arxiv_id_main for col in collections.collections):
                raise ValueError(f"Collection {self.arxiv_id_main} does not exist")

            # Use run_in_executor to run synchronous embedding in a thread
            loop = asyncio.get_event_loop()
            query_vector = await loop.run_in_executor(
                None, 
                self.embeddings.embed_query, 
                self.query
            )
            
            logger.info(f"Searching in collection: {self.arxiv_id_main}")
            response = self.client.search(
                collection_name=self.arxiv_id_main,
                query_vector=query_vector,
                limit=3  # Top 3 most relevant chunks
            )
            
            if not response:
                logger.warning("No results found in search")
                return ["No relevant information found for the query."]
                
            # Extract texts from response
            return [doc.payload.get("text", "") for doc in response]

        except ResponseHandlingException as e:
            logger.error(f"Qdrant connection error: {str(e)}")
            raise RuntimeError("Failed to connect to the database. Please check if Qdrant server is running.")
            
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            raise RuntimeError(f"Error retrieving documents: {str(e)}")

    async def chat_response(self) -> str:
        """
        Generate a chat response using retrieved context.
        
        :return: Generated response from LLM
        """
        try:
            # Retrieve relevant document chunks
            retrieved_chunks = await self.retrieve()
            chunks_text = "\n".join(retrieved_chunks)  # Combine chunks into a single string
            
            # Prepare prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an excellent researcher who can generate precise and informative responses based on the query, retrieved context, and chat history."),
                ("user", "Query: {query}\nRetrieved Chunks: {chunks}\nChat History: {history}\n\nGenerate a comprehensive and contextually relevant response.Only give relevant information and be conscise and try to give respose in bullets."),
            ])
            
            # Create chain and invoke
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "query": self.query,
                "chunks": chunks_text,
                "history": self.chat_history
            })
            
            return response.content
        
        except Exception as e:
            logger.error(f"Error in chat_response: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again later."