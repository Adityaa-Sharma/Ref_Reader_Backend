import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any

class Retrieval:
    def __init__(self, query: str, session_id: str, chat_history: str = ""):
        """
        Initialize Retrieval class with query, session ID, and optional chat history.
        
        :param query: User's current query
        :param session_id: Unique identifier for the current session/collection
        :param chat_history: Previous conversation context (optional)
        """
        load_dotenv()  
        
        # Qdrant client configuration
        self.client = QdrantClient(
            host=os.getenv('QDRANT_HOST', 'localhost'), 
            port=int(os.getenv('QDRANT_PORT', 6333))
        )
        
        # OpenAI embeddings configuration
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Language model configuration
        self.llm = ChatOpenAI(
            temperature=0, 
            model='gpt-3.5-turbo',
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Instance variables
        self.query = query
        self.session_id = session_id
        self.chat_history = chat_history

    async def retrieve(self) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from Qdrant collection.
        
        :return: List of retrieved document chunks
        """
        try:
            # Encode query and search in Qdrant
            query_vector = self.embeddings.embed_query(self.query)
            response = self.client.search(
                collection_name=self.session_id,
                query_vector=query_vector,
                limit=5  # Top 5 most relevant chunks
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Error retrieving documents: {str(e)}")

    async def chat_response(self) -> str:
        """
        Generate a chat response using retrieved context.
        
        :return: Generated response from LLM
        """
        # Retrieve relevant document chunks
        retrieved_chunks = await self.retrieve()
        
        # Prepare prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an excellent researcher who can generate precise and informative responses based on the query, retrieved context, and chat history."),
            ("user", "Query: {query}\nRetrieved Chunks: {chunks}\nChat History: {history}\n\nGenerate a comprehensive and contextually relevant response."),
        ])
        
        # Create chain and invoke
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "query": self.query,
            "chunks": retrieved_chunks,
            "history": self.chat_history
        })
        
        return response.content