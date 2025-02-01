import json
from dotenv import load_dotenv,find_dotenv,set_key
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

env_path = find_dotenv()
load_dotenv(env_path, override=True)

class PaperName:
    def __init__(self, query: str, citations: str, chat_history: str):
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("LLM_MODEL_NAME"),
            # model=os.getenv("AZURE_LLM_MODEL"),
            api_version="2024-08-01-preview",
            model_kwargs={
                "response_format": { "type": "text" }
            }
 
        )
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
            logger.debug(f"Processing query: {self.query[:100]}...")
            result = await self.chain.ainvoke({
                "query": self.query,
                "citations": self.citations,
                "chat_history": self.chat_history
            })
            
            if not result or not result.strip():
                logger.warning("Empty response from LLM")
                return {
                    "paper_name": "Unknown Paper",
                    "authors": "",
                    "arxiv_id": ""
                }

            logger.debug(f"Raw API response: {result[:200]}...")
            
            try:
                # Clean the response string
                result = result.strip()
                if result.startswith("```json"):
                    result = result[7:-3]  # Remove ```json and ``` markers
                elif result.startswith("{"):
                    # Already JSON formatted
                    pass
                else:
                    logger.warning(f"Unexpected response format: {result[:50]}...")
                
                result_dict = json.loads(result)
                logger.debug(f"Parsed result: {result_dict}")
                return {
                    "paper_name": result_dict.get("paper_name", "Unknown Paper"),
                    "authors": result_dict.get("authors", ""),
                    "arxiv_id": result_dict.get("arxiv_id", "")
                }
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}, Raw result: {result[:200]}")
                return {
                    "paper_name": "Error in paper identification",
                    "authors": "",
                    "arxiv_id": ""
                }
                
        except Exception as e:
            logger.error(f"Error in get_paper_name: {str(e)}")
            raise Exception(f"Error getting paper name: {str(e)}")