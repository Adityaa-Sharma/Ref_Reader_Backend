import os
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI,AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Union, Any

class Retrieval:
    def __init__(self, query: Union[str, Any], arxiv_id :str, chat_history: str = ""):
        load_dotenv()  
        
        # Ensure query is a string
        self.query = str(query) if query is not None else ""
        
        # Qdrant client configuration
        self.client = QdrantClient(
            host=os.getenv('QDRANT_HOST', 'localhost'), 
            port=int(os.getenv('QDRANT_PORT', 6333))
        )
        
        # OpenAI embeddings configuration
        # self.embeddings = OpenAIEmbeddings(
        #     model="text-embedding-3-small",
        #     api_key=os.getenv('OPENAI_API_KEY')
        # )
        self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                deployment=os.getenv("embedding_deployment"),
                model=os.getenv("AZURE_EMBEDDING_MODEL"),
                api_version="2024-03-01-preview"
            )
        # Language model configuration
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment=os.getenv("llm_deployment"),
            model=os.getenv("AZURE_LLM_MODEL")
            
        )
        self.arxiv_id = arxiv_id
        
        # Instance variables
        # self.session_id = session_id
        self.chat_history = chat_history

    async def retrieve(self) -> List[str]:
        """
        Asynchronously retrieve relevant documents from Qdrant collection.
        
        :return: List of retrieved document texts
        """
        try:
            # Use run_in_executor to run synchronous embedding in a thread
            loop = asyncio.get_event_loop()
            query_vector = await loop.run_in_executor(
                None, 
                self.embeddings.embed_query, 
                self.query
            )
            
            # Search in Qdrant
            response = self.client.search(
                collection_name=self.arxiv_id,
                query_vector=query_vector,
                limit=3  # Top 3 most relevant chunks
            )
                
            # Extract texts from response
            return [doc.payload.get("text", "") for doc in response]
        except Exception as e:
            # Log the full exception for debugging
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error retrieving documents: {str(e)}")

    async def chat_response(self) -> str:
        """
        Generate a chat response using retrieved context.
        
        :return: Generated response from LLM
        """
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