import json
from dotenv import load_dotenv,find_dotenv,set_key
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

env_path = find_dotenv()

load_dotenv(env_path, override=True)

class PaperName:
    def __init__(self, query: str, citations: str, chat_history: str):
        self.llm = ChatOpenAI(temperature=0)
        self.query = query
        self.citations = citations
        self.chat_history = chat_history

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an excellent researcher who can identify paper names from queries and citations adn preious cht history."),
            ("user", """identify the paper name that user is refering in the query from the citations and also consider chat_history is needed.
                      Query: {query},
            Citations: {citations},
            Chat History: {chat_history}
                      Return the details in this format: {{"paper_name": "paper name/title", "authors": "authors name", "arxiv_id": "arxiv id"}}
                      Note: Make sure to return a valid JSON string.""")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def get_paper_name(self):
        """Async version of get_paper_name"""
        try:
            result = await self.chain.ainvoke({
                "query": self.query,
                "citations": self.citations,
                "chat_history": self.chat_history
            })  
            # Parse the string result into a dictionary
            result_dict = json.loads(result)
            return {
                "paper_name": result_dict.get("paper_name", ""),
                "authors": result_dict.get("authors", ""),
                "arxiv_id": result_dict.get("arxiv_id", "")
            }
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing JSON response: {str(e)}")
        except Exception as e:
            raise Exception(f"Error getting paper name: {str(e)}")