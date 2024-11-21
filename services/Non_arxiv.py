from services.Agents import analyze_research_paper
import json

class NonArxiv:
    def __init__(self, query: str, citations: str):
        self.query = query
        self.citations = citations

    def get_paper_details(self):
        # Analyze the research paper
        result = analyze_research_paper(self.query)
        return result