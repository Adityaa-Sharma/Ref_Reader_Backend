from services.Agents import analyze_research_paper
# from test import run_research
from services.websearch import run_research
import json

class NonArxiv:
    def __init__(self, query: str, citations: str):
        self.query = query
        self.citations = citations

    def get_paper_details(self):
        # Analyze the research paper
        # result = analyze_research_paper(self.query)
        result = run_research(self.query)
        return result