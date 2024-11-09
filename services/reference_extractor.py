import re
import logging
import fitz
import os



# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler which logs even debug messages
fh = logging.FileHandler('logs/app.log')
fh.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)




class ReferenceExtractor:

    @staticmethod
    def extract_references_from_text(text):
        reference_pattern = re.compile(r'\[\d+\]')
        references = reference_pattern.findall(text)
        return references

    @staticmethod
    def extract_bibliography(text):
        bibliography_pattern = re.compile(r'(References|Bibliography)(.*)', re.DOTALL | re.IGNORECASE)
        match = bibliography_pattern.search(text)
        if match:
            return match.group(2)
        return ""

    @staticmethod
    def extract_reference_citations(bibliography_text):
        citations = {}
        citation_pattern = re.compile(r'\[(\d+)\](.*?)(?=\[\d+\]|\Z)', re.DOTALL)
        for match in citation_pattern.finditer(bibliography_text):
            ref_number = match.group(1)
            citation_text = match.group(2).strip()
            
            logger.debug(f"Processing citation {ref_number}: {citation_text}")
            
            citation_text = ' '.join(citation_text.split())
            
            # Basic validation for citation content
            if re.search(r'[“"“"](.*?)[”"”"]', citation_text) and re.search(r'\b[A-Z][a-zA-Z]*\b', citation_text):
                citations[ref_number] = {'reference': citation_text}
            else:
                logger.debug(f"Skipping incomplete citation {ref_number}")
        
        return citations

    @staticmethod
    def document_loader(file_path):
        # Load PDF and extract all text
        doc = fitz.open(file_path)
        all_text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            all_text += page.get_text()
        
        logger.debug(f"Extracted text from PDF: {all_text[:500]}...")  # Log first 500 characters
        
        # Extract references and bibliography text
        references = ReferenceExtractor.extract_references_from_text(all_text)
        bibliography_text = ReferenceExtractor.extract_bibliography(all_text)
        logger.debug(f"Extracted bibliography: {bibliography_text[:500]}...")  # Log first 500 characters
        
        # Extract reference citations
        citations = ReferenceExtractor.extract_reference_citations(bibliography_text)
        
        return references, citations

    
## test
references, citations = ReferenceExtractor.document_loader('C:\\Users\\91978\\Desktop\\Ref_reader_backend\\encoder decoder.pdf')

