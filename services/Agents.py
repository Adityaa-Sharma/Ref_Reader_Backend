from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
import json

search_tool = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Creating specialized agents
researcher = Agent(
    role='Research Analyst',
    goal='Thoroughly analyze research papers and extract key information',
    backstory="""You are an experienced research analyst with expertise in 
    reading and analyzing academic papers. Your strength lies in quickly 
    understanding complex research methodologies and findings.""",
    tools=[search_tool, wikipedia],
    verbose=True
)

summarizer = Agent(
    role='Content Summarizer',
    goal='Create clear and concise summaries of complex research information',
    backstory="""You are a skilled content summarizer who can break down 
    complex academic concepts into easily understandable summaries. You excel 
    at identifying key points and main contributions.""",
    tools=[search_tool],
    verbose=True
)

critic = Agent(
    role='Research Critic',
    goal='Evaluate research methodology and findings critically',
    backstory="""You are a critical thinker with extensive experience in 
    peer review. You analyze research papers for potential limitations, 
    biases, and areas of improvement.""",
    tools=[search_tool],
    verbose=True
)

def analyze_research_paper(paper_query):
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

    # Creating and running the crew
    crew = Crew(
        agents=[researcher, summarizer, critic],
        tasks=[research_task, summary_task, critique_task],
        verbose=True
    )

    result = crew.kickoff()
    
    # Convert the CrewOutput to a serializable format
    try:
        serializable_result = {
            "research_analysis": str(result),  # Convert the entire output to string
        }
        return json.dumps(serializable_result)
    except Exception as e:
        print(f"Error processing results: {e}")
        return json.dumps({"error": f"Failed to process results: {str(e)}"})
# Example usage
# if __name__ == "__main__":
#     paper_query = "Attention Is All You Need - Transformer paper"
#     results = analyze_research_paper(paper_query)
#     print(results)