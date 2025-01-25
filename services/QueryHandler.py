import json
from dotenv import load_dotenv,find_dotenv,set_key
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

class QueryHandler:
    def __init__(self, query: str, paper_name: str, chat_history: str, citations: str):
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment=os.getenv("llm_deployment"),
            model=os.getenv("AZURE_LLM_MODEL"),
            api_version="2024-03-01-preview"
            
            )
        self.query = query
        self.paper_name = paper_name
        self.chat_history = chat_history
        self.citations = citations

    async def query_rephraser(self):
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an excellent researcher who can rephrase the query according to the paper name and previous chat history."),
            ("user", f'''Rephrase the query: {self.query} , according to the paper name: {self.paper_name} and previous chat history: {self.chat_history}
                         always refer to chat_history for comparison related query rephrasing'''),
        ])

        # Set the input for the chain, ensuring the required variables are provided
        prompt_input = {
            "query": self.query,
            "citations": self.citations,
            "chat_history": self.chat_history
        }

        # Invoke the chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        response = await self.chain.ainvoke(input=prompt_input)
        
        return response
