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
from database.database import save_pdf,get_pdf_citations, get_session_id,get_paper,save_paper, get_chat_history,save_chat_history,check_if_MainPaper_already_processed,save_MainPaper,AlreadyProcessed,SaveInProcessedPapers
import uuid
import json
import os
import logging
from logging.handlers import RotatingFileHandler
import asyncio
from concurrent.futures import ThreadPoolExecutor
from mangum import Mangum   
import requests
from fastapi import Form, Body, Request
from typing import Optional
from logs.log import setup_logging


# Logging configuration function


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
class FileInput(BaseModel):
    paper_title: str = None
    arxiv_id: str = None

class ArxivInput(BaseModel):
    arxiv_id: str

@app.post("/extract_references/arxiv")
async def extract_references_arxiv(arxiv_input: ArxivInput):
    temp_file = None
    try:
        arxiv_id = arxiv_input.arxiv_id
        # pdf_name = f"{arxiv_id}.pdf"
        
        # Check if already processed
        pdf_entry = check_if_MainPaper_already_processed(arxiv_id)
        if (pdf_entry):
            session_id=str(uuid.uuid4())
            return JSONResponse(content={
                "session_id": session_id,
                "arxiv_id": arxiv_id}
            )

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        temp_file.close()

        # Download PDF
        response = requests.get(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download arxiv PDF")
        
        # Save PDF content
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)

        # Process PDF
        citations = await asyncio.to_thread(ReferenceExtractor.document_loader, temp_file_path)
        if not citations:
            raise HTTPException(status_code=400, detail="No citations found in PDF")

        # Save to database
        session_id = str(uuid.uuid4())
        save_MainPaper(
            arxiv_id=arxiv_id,
            citations=json.dumps(citations)
        )
        return JSONResponse(content={
            "session_id": session_id,
            "arxiv_id": arxiv_id
        })

    except Exception as e:
        logger.error(f"Error in extract_references_arxiv: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.close(os.open(temp_file_path, os.O_RDONLY))
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")

@app.post("/extract_references/upload")
async def extract_references_upload(
    file: UploadFile = File(...),
    paper_title: str = Form(...)
):
    temp_file = None
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Check if already processed
        pdf_entry = get_session_id(paper_title)
        if pdf_entry:
            return JSONResponse(content={
                "session_id": pdf_entry,
                "saved_id": {"title": paper_title}
            })

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        temp_file.close()

        # Save uploaded content
        content = await file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)

        # Process PDF
        citations = await asyncio.to_thread(ReferenceExtractor.document_loader, temp_file_path)
        if not citations:
            raise HTTPException(status_code=400, detail="No citations found in PDF")

        # Save to database
        session_id = str(uuid.uuid4())
        save_pdf(
            session_id=session_id,
            pdf_name=paper_title,
            citations=json.dumps(citations)
        )

        return JSONResponse(content={
            "session_id": session_id,
            "saved_id": {"title": paper_title}
        })

    except Exception as e:
        logger.error(f"Error in extract_references_upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.close(os.open(temp_file_path, os.O_RDONLY))
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")

class Chat(BaseModel):
    response: dict
    query: str    

@app.get("/chat/")
async def chat(
    response: str,  # Will receive JSON string
    query: str
):
    try:
        # Parse the response JSON string
        response_dict = json.loads(response)
        session_id = response_dict.get('session_id')
        arxiv_id_main = response_dict.get('arxiv_id')
        
        if not session_id or not arxiv_id_main:
            raise HTTPException(
                status_code=400, 
                detail="Invalid response format. Must include session_id and arxiv_id"
            )

        pdf_entry = get_pdf_citations(arxiv_id_main)
        
        if not pdf_entry:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        chat_history = await get_chat_history(session_id)
        
        paper_content = await PaperName(query, str(pdf_entry["citations"]), chat_history).get_paper_name()
        
        logger.debug(f"Paper content arxiv_id: {paper_content.get('arxiv_id')}")

        # Check if arxiv_id exists and is valid
        if paper_content.get('arxiv_id') and ':' in paper_content['arxiv_id']:
            arxiv_id = paper_content['arxiv_id'].split(':')[1]
            
            # Check if paper already exists
            if not AlreadyProcessed(arxiv_id_main, arxiv_id):
                # Process new paper
                
                
                ingestor = VectorIngestor(
                    paper_name=paper_content['paper_name'],
                    main_arxiv_id=arxiv_id_main,
                    arxiv_id=arxiv_id
                )
                await ingestor.arxiv_handling(arxiv_id)
                SaveInProcessedPapers(arxiv_id_main, arxiv_id)
            
            # Handle query for both new and existing papers
            query_handler = QueryHandler(query, paper_content['paper_name'], chat_history, pdf_entry["citations"])
            rephrased_query = await query_handler.query_rephraser()
            logger.debug(f"Rephrased query: {rephrased_query}")
            
            Retriever = Retrieval(rephrased_query, arxiv_id,arxiv_id_main, chat_history)
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

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid response JSON format")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)









