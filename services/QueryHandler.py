import json
from dotenv import load_dotenv,find_dotenv,set_key
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()


class QueryHandler:
    def __init__(self,query:str, paper_name:str,)