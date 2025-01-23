from fastapi import FastAPI, File, UploadFile,HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from services.reference_extractor import ReferenceExtractor
from services.paper_name import PaperName
from services.Ingestion import VectorIngestor
from pydantic import BaseModel
from services.Non_arxiv import NonArxiv
from services.QueryHandler import QueryHandler
from services.Retrieval import Retrieval
import tempfile
from database.database import save_pdf,get_pdf_citations, get_session_id,get_paper,save_paper, get_chat_history,save_chat_history
import uuid
import json
import os
import logging
from logging.handlers import RotatingFileHandler
import asyncio
from concurrent.futures import ThreadPoolExecutor
from mangum import Mangum   

logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Logging configuration function
def setup_logging():
    # Log file path
    log_file_path = os.path.join(logs_dir, 'app.log')

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all log levels

    # Create a file handler
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5  # Keep 5 backup files
    )
    file_handler.setLevel(logging.DEBUG)  # Set to capture DEBUG and above
    file_handler.setLevel(logging.INFO)  # File only shows INFO and above

    # Create a formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Optional: Add console handler for immediate feedback during development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Console only shows INFO and above
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Setup logging
logger = setup_logging()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]
)

@app.post("/extract_references/")
async def extract_references(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name        
    try:
        pdf_name = file.filename
        pdf_entry = get_session_id(pdf_name)
        if pdf_entry:   
            return JSONResponse(content={"session_id": pdf_entry})
        
        citations = ReferenceExtractor.document_loader(temp_file_path)
        session_id = str(uuid.uuid4())
        save_pdf(session_id, pdf_name, json.dumps(citations))
        return JSONResponse(content={"session_id": session_id})

    except Exception as e:
       
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.unlink(temp_file_path)
        
class Chat(BaseModel):
    session_id: str
    query: str    

@app.get("/chat/")
async def chat(session_id: str, query: str):
    pdf_entry = get_pdf_citations(session_id)
    
    if not pdf_entry:
        raise HTTPException(status_code=404, detail="Session ID not found")
    
    chat_history = await get_chat_history(session_id)
    
    paper_content = await PaperName(query, str(pdf_entry["citations"]), chat_history).get_paper_name()
    
    print("arxiv_id", paper_content.get('arxiv_id'))

    # Check if arxiv_id exists and is valid
    if paper_content.get('arxiv_id') and ':' in paper_content['arxiv_id']:
        arxiv_id = paper_content['arxiv_id'].split(':')[1]
            
        # Check if paper already exists
        if get_paper(arxiv_id) is None:
            # Save new paper
            save_paper(
                session_id=session_id,
                paper_name=paper_content['paper_name'],
                authors=paper_content['authors'],
                arxiv_id=arxiv_id
            )
            
            # Initialize and process with vector ingestor
            ingestor = VectorIngestor(
                session_id=session_id,
                paper_name=paper_content['paper_name'],
                arxiv_id=arxiv_id
            )
            
            ingest_result = await ingestor.arxiv_handling(arxiv_id)
            
            # Handle query
            query_handler = QueryHandler(query, paper_content['paper_name'], chat_history, pdf_entry["citations"])
            rephrased_query = await query_handler.query_rephraser()
            print("rephrased_query", rephrased_query)
            Retriever = Retrieval(rephrased_query, session_id, chat_history)
            response = await Retriever.chat_response()
            await save_chat_history(session_id, rephrased_query, response)
            return JSONResponse(content={"response": response})
    
        else:
            # Handle existing paper with retriever
            query_handler = QueryHandler(query, paper_content['paper_name'], chat_history, pdf_entry["citations"])
            rephrased_query = await query_handler.query_rephraser()
            print("rephrased_query", rephrased_query)
            Retriever = Retrieval(rephrased_query, session_id, chat_history)
            response = await Retriever.chat_response()
            await save_chat_history(session_id, rephrased_query, response)
            return JSONResponse(content={"response": response})

    else:
        # Non-arxiv block
        paper_name = paper_content['paper_name']
        query_handler = QueryHandler(query, paper_name, chat_history, pdf_entry["citations"])
        rephrased_query = await query_handler.query_rephraser()
        NonArxivHandler = NonArxiv(rephrased_query, paper_name)
        print("rephrased_query", rephrased_query)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(
                pool, NonArxivHandler.get_paper_details
                )
        response = json.loads(response)
        await save_chat_history(session_id, rephrased_query, response)


    return JSONResponse(content={"response": response.get("research_analysis")})

    


handler = Mangum(app)
 
 



        



