import json
from dotenv import load_dotenv,find_dotenv,set_key
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core import ChtPromptTemplate
import os

load_dotenv()


class QueryHandler:
    def __init__(self,query:str, paper_name:str,chat_history:str):
        self.llm = ChatOpenAI(temperature=0)
        self.query = query
        self.paper_name = paper_name
        self.chat_history=chat_history
        
    async def query_rephraser(self):
        self.prompt=ChtPromptTemplate.from_messages([
            ("system","You are an excellent researcher who can rephrase the query according to the paper name and previous chat History."),
            ("user",f"Rephrase the query: {self.query} , according to the paper name: {self.paper_name} and previous chat history: {self.chat_history}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
        return self.chain.ainvoke()
    
    
        