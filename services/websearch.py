from crewai import Agent, Task, Crew
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Set SERPAPI_API_KEY
os.environ["SERPAPI_API_KEY"] = "Your serpapi key here"

# Define tool input schemas
class SearchToolInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(..., description="The search query to look up.")

class WikipediaToolInput(BaseModel):
    """Input schema for Wikipedia tool."""
    query: str = Field(..., description="The topic to look up on Wikipedia.")

# Define CrewAI compatible tools
class SearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Useful for searching the internet for recent or specific information."
    args_schema: type[BaseModel] = SearchToolInput
    
    def _run(self, query: str) -> str:
        try:
            search = SerpAPIWrapper(serpapi_api_key=os.environ["SERPAPI_API_KEY"])
            result = search.run(query)
            return result if result else "No results found"
        except Exception as e:
            return f"Search failed: {str(e)}"

class WikipediaTool(BaseTool):
    name: str = "Wikipedia Search"
    description: str = "Useful for getting detailed information from Wikipedia articles."
    args_schema: type[BaseModel] = WikipediaToolInput
    
    def _run(self, query: str) -> str:
        try:
            wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            result = wiki.run(query)
            return result if result else "No Wikipedia entry found"
        except Exception as e:
            return f"Wikipedia search failed: {str(e)}"

# Initialize tools
tools = [SearchTool(), WikipediaTool()]

# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    temperature=0.7,
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    deployment_name=f"azure/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}"
)

# Create the research agent
researcher = Agent(
    role='Research Assistant',
    goal='Conduct thorough research on given topics using internet search and Wikipedia',
    backstory="""You are a skilled research assistant with expertise in gathering and 
    synthesizing information from multiple sources. You excel at combining data from 
    web searches and Wikipedia to provide comprehensive insights.""",
    tools=tools,
    llm=llm,
    verbose=True
)

def run_research(query: str):
    # Create a research task
    research_task = Task(
        description=f"""Research the following topic thoroughly: {query}
        1. Search the internet for current information
        2. Check Wikipedia for background information
        3. Combine the information into a comprehensive summary
        4. Include sources for your findings
        5. If you do not find exact information, provide the closest match.""",
        expected_output="""A comprehensive research summary including:
        - Current information from web searches
        - Background information from Wikipedia
        - Combined analysis of findings
        - Source citations and references""",
        agent=researcher
    )

    # Create and run the crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )

    result = crew.kickoff()
    return json.dumps({"research_analysis": str(result)})

# if __name__ == "__main__":
#     # Test the research capability
#     query = "What are the latest developments in quantum computing?"
#     result = run_research(query)
#     print("\nResearch Results:")
#     print(result)
