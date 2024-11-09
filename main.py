from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from services.reference_extractor import ReferenceExtractor
import logging
from tempfile import NamedTemporaryFile
import shutil


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-references")
async def extract_references(file: UploadFile = File(...)):
    temp_file = NamedTemporaryFile(delete=False)
    with open(temp_file.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    references=ReferenceExtractor.document_loader(temp_file.file)
    return references



##     
        



