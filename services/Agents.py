from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_openai import AzureChatOpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm_config():
    try:
        return {
            "llm": AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                deployment_name=os.getenv("LLM_MODEL_NAME"),
                api_version=os.getenv("API_VERSION", "2024-08-01-preview"),
                temperature=0.7,
                model_kwargs={
                    "response_format": { "type": "text" }
                }
            ),
            "verbose": True
        }
    except Exception as e:
        print(f"Error in LLM configuration: {e}")
        return None

search_tool = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Modified agent creation with error handling
def create_agent(role, goal, backstory, tools):
    try:
        config = get_llm_config()
        if not config:
            raise Exception("Failed to get LLM configuration")
            
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            **config
        )
    except Exception as e:
        print(f"Error creating agent {role}: {e}")
        return None

# Initialize agents with error handling
researcher = create_agent(
    role='Research Analyst',
    goal='Thoroughly analyze research papers and extract key information',
    backstory="""You are an experienced research analyst with expertise in 
    reading and analyzing academic papers. Your strength lies in quickly 
    understanding complex research methodologies and findings.""",
    tools=[search_tool, wikipedia]
)

summarizer = create_agent(
    role='Content Summarizer',
    goal='Create clear and concise summaries of complex research information',
    backstory="""You are a skilled content summarizer who can break down 
    complex academic concepts into easily understandable summaries. You excel 
    at identifying key points and main contributions.""",
    tools=[search_tool]
)

critic = create_agent(
    role='Research Critic',
    goal='Evaluate research methodology and findings critically',
    backstory="""You are a critical thinker with extensive experience in 
    peer review. You analyze research papers for potential limitations, 
    biases, and areas of improvement.""",
    tools=[search_tool]
)

def analyze_research_paper(paper_query):
    try:
        # [Previous task definitions remain the same]
        research_task = Task(
            description=f"""Research and gather detailed information about the paper: {paper_query}.
            Focus on:
            1. Main research questions
            2. Methodology used
            3. Key findings
            4. Data collection methods
            5. Technical implementation details
            Provide specific quotes and references from the paper.""",
            expected_output="""A detailed analysis of the research paper including main questions,
            methodology, findings, data collection methods, and technical details with specific quotes.""",
            agent=researcher
        )

        summary_task = Task(
            description=f"""Create a comprehensive yet concise summary of the research paper.
            Include:
            1. Main objectives
            2. Key contributions
            3. Important results
            4. Practical implications
            Make it accessible for a general technical audience.""",
            expected_output="""A clear and concise summary of the research paper covering objectives,
            contributions, results, and practical implications, written for a technical audience.""",
            agent=summarizer
        )

        critique_task = Task(
            description=f"""Provide a critical analysis of the research paper.
            Evaluate:
            1. Methodology robustness
            2. Validity of conclusions
            3. Potential limitations
            4. Future research directions
            5. Practical applicability
            Support your critique with specific examples from the paper.""",
            expected_output="""A detailed critical analysis of the research paper covering methodology,
            conclusions, limitations, future directions, and practical applications, with specific examples.""",
            agent=critic
        )

        # Check if agents were created successfully
        if not all([researcher, summarizer, critic]):
            raise Exception("Failed to initialize one or more agents")

        # Creating and running the crew
        crew = Crew(
            agents=[researcher, summarizer, critic],
            tasks=[research_task, summary_task, critique_task],
            verbose=True
        )

        result = crew.kickoff()
        
        # Convert the CrewOutput to a serializable format
        return json.dumps({"research_analysis": str(result)})
    except Exception as e:
        error_message = f"Error during analysis: {str(e)}"
        print(error_message)
        return json.dumps({
            "error": error_message,
            "status": "failed",
            "suggestion": "Please check your API credentials and billing status"
        })

# Example usage
if __name__ == "__main__":
    paper_query = "Attention Is All You Need - Transformer paper"
    results = analyze_research_paper(paper_query)
    print(results)