# Ref_Reader_Backend

An AI-powered research paper analysis system using CrewAI agents and vector search capabilities.

## Features

### Advanced Search Capabilities
- Google Scholar search via SerperDev API
- Vector-based semantic search with Qdrant
- Wikipedia integration for background information
- Web search for supplementary information

### Multi-Agent Research System
- Research Analyst agent for scholarly content analysis
- Research Critic agent for methodology evaluation
- Combined analysis from multiple sources

### Document Processing
- Vector embeddings via Azure OpenAI
- Async document retrieval
- Context-aware response generation

## Setup

### Prerequisites
- Python 3.8+
- Docker (for Qdrant)
- Azure OpenAI API access
- SerperDev API key

### Environment Variables
Create a `.env` file with:
```bash
# Azure OpenAI Configuration
AZURE_API_BASE=your_azure_endpoint
AZURE_API_KEY=your_azure_key
AZURE_API_VERSION=your_api_version
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
LLM_MODEL_NAME=your_model_name

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# SerperDev API
SERPAPI_API_KEY=your_serper_api_key
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Ref_Reader_backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### Vector Search
```python
from services.Retrieval import Retrieval

# Initialize retriever
retriever = Retrieval(
    query="your research query",
    arxiv_id="paper_id",
    arxiv_id_main="collection_name"
)

# Get search results
results = await retriever.retrieve()
```

### Research Analysis
```python
from services.Agents import analyze_research_paper

# Analyze a research paper
results = analyze_research_paper("Attention Is All You Need - Transformer paper")
```

### Web Search
```python
from services.websearch import run_research

# Perform web research
results = run_research("Latest developments in quantum computing")
```

## API Reference

### Retrieval Class
- `retrieve()`: Performs vector search in Qdrant
- `chat_response()`: Generates context-aware responses

### Agents
- `analyze_research_paper()`: Comprehensive paper analysis
- `run_research()`: General web research

### Tools
- `ScholarSearchTool`: Google Scholar search
- `WebSearchTool`: General web search
- `WikipediaTool`: Wikipedia lookups

## Error Handling

The system includes comprehensive error handling:
- Connection verification for Qdrant
- API credential validation
- Fallback options for searches
- Detailed error logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Dependencies

Main dependencies:
- `crewai`: Multi-agent orchestration
- `langchain`: LLM framework
- `qdrant-client`: Vector database client
- `azure-openai`: Azure OpenAI integration

## License

MIT License

## Contact

For support or queries, please open an issue in the repository.