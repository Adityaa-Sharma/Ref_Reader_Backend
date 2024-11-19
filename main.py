from fastapi import FastAPI, File, UploadFile,HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from services.reference_extractor import ReferenceExtractor
from services.paper_name import PaperName
from services.Ingestion import VectorIngestor
from pydantic import BaseModel
from services.Agents import NonArxiv
from services.QueryHandler import QueryHandler
from services.Retrieval import Retrieval
import tempfile
from database.database import save_pdf, get_db_cursor,get_pdf_citations, get_session_id,get_paper,save_paper, get_chat_history,save_chat_history
import uuid
import json
import os


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    
    paper_content = await PaperName(query, str(pdf_entry["citations"])).get_paper_name()
    arxiv_id = paper_content['arxiv_id'].split(':')[1] if paper_content['arxiv_id'] != "" else ""
    if paper_content['arxiv_id']:
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
                
                ## retriever will come here
                query_handler = QueryHandler(query, paper_content['paper_name'], get_chat_history(session_id))
                rephrased_query = await query_handler.query_rephraser()
                Retriever=Retrieval(rephrased_query,session_id,get_chat_history(session_id))
                response = await Retriever.chat_response()
                ## save query and response in chat history
                save_chat_history(session_id, rephrased_query, response)
    
     
            else:
                ## only retriever will come here
                query_handler = QueryHandler(query, paper_content['paper_name'], get_chat_history(session_id))
                rephrased_query = await query_handler.query_rephraser()
                Retriever=Retrieval(rephrased_query,session_id,get_chat_history(session_id))
                response = await Retriever.chat_response()
                save_chat_history(session_id, rephrased_query, response)

    else:
        paper_name = paper_content['paper_name']
        NonArxivHandler = NonArxiv(query,paper_name)
        

    

   
    
    
    


 
 
 
 



        



