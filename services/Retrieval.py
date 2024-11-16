import openai
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()



class Retrieval:
    def __init__(self,query:str,session_id:str,chat_history:str):
        self.query = query
        self.session_id = session_id
        self.chat_history=chat_history
        self.client = QdrantClient(host='localhost', port=6333)
        self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"  
            )
        self.llm=ChatOpenAI(temperature=0)
        
    async def retrieve(self):
        try:
            response = self.client.search(
                collection_name=self.session_id,
                query_vector=self.embeddings.encode(self.query),
                top=5
            )
            return response
        except Exception as e:
            raise Exception(f"Error retrieving papers: {str(e)}")
    
    async def chat_response(self):
        retrived_chunks = await self.retrieve()
        ## giving the chat histy , quer and retrievde chunks to the llm
        self.prompt=ChatPromptTemplate.from_messages([
            ("system","You are an excellent researcher who can generate responses based on the query, retrieved chunks and chat history."),
            ("user",f"Generate a chat response based on the query: {self.query}, retrieved chunks: {retrived_chunks} and chat history: {self.chat_history}")
        ])
        self.chain = self.prompt | self.llm
        return self.chain.ainvoke() 
    
        
        
    
    
        
        
        