# Retriever class

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv()
load_dotenv(env_path, override=True)

class Retriever:
    def __init__(self, query: str, citations: str):
        self.llm = ChatOpenAI(temperature=0)
        self.query = query
        self.citations = citations

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an excellent researcher who can identify paper names from queries and citations."),
            ("user", """identify the paper name that user is refering in the query from the citations.
                      Query: {query},
            Citations: {citations}
                      Return the details in this format: {{"paper_name": "paper name/title", "authors": "authors name", "arxiv_id": "arxiv id"}}
                      Note: Make sure to return a valid JSON string.""")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()