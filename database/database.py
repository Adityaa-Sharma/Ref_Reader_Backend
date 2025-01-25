import psycopg2
from psycopg2 import sql
import contextlib
from contextlib import contextmanager
import uuid
import asyncio
from typing import Union, Any
import json

# Database connection parameters
db_params = {
    "user": "postgres.qyfbggcqowfbryiifibd",
    "password": "adityasharma@123",
    "host": "aws-0-ap-south-1.pooler.supabase.com",
    "port": "6543",
    "database": "postgres"
}


@contextlib.contextmanager
def get_db_connection():
    conn = psycopg2.connect(**db_params)
    try:
        yield conn
    finally:
        conn.close()

@contextlib.contextmanager
def get_db_cursor(commit=False):
    with get_db_connection() as connection:
        cursor = connection.cursor()
        try:
            yield cursor
            if commit:
                connection.commit()
        finally:
            cursor.close()

def create_tables():
    with get_db_cursor(commit=True) as cur:
        # Create sessions table with UUID
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                session_id UUID UNIQUE NOT NULL
            )
        """)
        

        
        
        ## creat table for paper , in this it will the , paper id (incremental),session id , paper name , authors , arxiv id(can be null)
        # cur.execute("""
        #     CREATE TABLE IF NOT EXISTS papers (
        #         id SERIAL PRIMARY KEY,
        #         session_id UUID REFERENCES sessions(session_id),
        #         paper_name VARCHAR NOT NULL,
        #         authors VARCHAR NOT NULL,
        #         arxiv_id VARCHAR
        #     )
        # """)
        cur.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            query TEXT NOT NULL,
            response JSONB NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    ''')
        # create atable to store the arxiv id of the papers that are being inputed
        cur.execute('''CREATE TABLE IF NOT EXISTS Main_Paper (
            id SERIAL PRIMARY KEY,
            arxiv_id VARCHAR NOT NULL,
            citations TEXT NOT NULL
        )''') 
        
        # create a table having the name ProcessedPapers, in which it has column main_paper_arxiv_id, and other columns is already_processed which is the list of the arxiv ids
    
        cur.execute('''CREATE TABLE IF NOT EXISTS ProcessedPapers (
            id SERIAL PRIMARY KEY,
            main_paper_arxiv_id VARCHAR NOT NULL UNIQUE,
            already_processed VARCHAR[] DEFAULT ARRAY[]::VARCHAR[]
        )''')


def save_pdf(session_id: str, pdf_name: str, citations: str, arxiv_id: str = None):
    with get_db_cursor(commit=True) as cur:
        # Create session if not exists
        cur.execute("SELECT session_id FROM sessions WHERE session_id = %s", (session_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO sessions (session_id) VALUES (%s)", (session_id,))
        
        # Check if PDF already exists
        cur.execute("SELECT id FROM pdfs WHERE session_id = %s AND pdf_name = %s", 
                   (session_id, pdf_name))
        if not cur.fetchone():
            # Insert PDF entry with arxiv_id
            cur.execute("""
                INSERT INTO pdfs (session_id, pdf_name, citations, arxiv_id)
                VALUES (%s, %s, %s, %s)
            """, (session_id, pdf_name, citations, arxiv_id))
            
            

def get_pdf_citations(arxiv_id):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT citations FROM main_paper WHERE arxiv_id = %s", (arxiv_id,))
        result = cursor.fetchone()
        if result:
            return { "citations": result[0]}
        return None

def get_session_id(pdf_name):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT session_id FROM pdfs Where pdf_name = %s", (pdf_name,))
        result = cursor.fetchone()
        if result:
            return result[0]
        return None
    
def save_paper(session_id: str, paper_name: str, authors: str, arxiv_id: str):
    with get_db_cursor(commit=True) as cur:
        cur.execute("""
            INSERT INTO papers (session_id, paper_name, authors, arxiv_id)
            VALUES (%s, %s, %s, %s)
        """, (session_id, paper_name, authors, arxiv_id))
        
## get check if the paper arxiv id is present in the papers table or not
def get_paper(arxiv_id):
    if(arxiv_id==""):
        return None
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM papers WHERE arxiv_id = %s", (arxiv_id,))
        result = cursor.fetchone()
        if result:
            return {
                "session_id": result[1],
                "paper_name": result[2],
                "authors": result[3],
                "arxiv_id": result[4]
            }
        return None
    
# check if the arxiv id is present in the main paper table or not
def check_if_MainPaper_already_processed(arxiv_id):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM main_Paper WHERE arxiv_id = %s", (arxiv_id,))
        result = cursor.fetchone()
        if result:
            return True
        return False

# saving hte main paper
def save_MainPaper(arxiv_id: str, citations: str):
    with get_db_cursor(commit=True) as cursor:
        cursor.execute("""
            INSERT INTO Main_Paper (arxiv_id, citations)
            VALUES (%s, %s)
        """, (arxiv_id, citations))
# want to check if the arxiv is present in th list of already processed paper corresponding to the main paper
def AlreadyProcessed(main_arxiv_id: str, arxiv_id: str) -> bool:
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM ProcessedPapers 
                WHERE main_paper_arxiv_id = %s 
                AND %s = ANY(already_processed)
            )
        """, (main_arxiv_id, arxiv_id))
        return cursor.fetchone()[0]
    
def SaveInProcessedPapers(main_arxiv_id: str, arxiv_id: str):
    with get_db_cursor(commit=True) as cursor:
        try:
            cursor.execute("""
                INSERT INTO ProcessedPapers (main_paper_arxiv_id, already_processed)
                VALUES (%s, ARRAY[%s])
                ON CONFLICT (main_paper_arxiv_id) DO
                UPDATE SET already_processed = array_append(ProcessedPapers.already_processed, %s)
                WHERE ProcessedPapers.main_paper_arxiv_id = %s
            """, (main_arxiv_id, arxiv_id, arxiv_id, main_arxiv_id))
        except Exception as e:
            print(f"Error saving processed paper: {str(e)}")
            raise
    



 
    
## insert the chat history in the chat_history table
import json
from typing import Union, Any
import asyncio

async def save_chat_history(session_id: str, query: Union[str, Any], response: Union[str, Any]):
    """
    Asynchronously save chat history to the database.
    
    :param session_id: The session identifier
    :param query: The user's query (can be a coroutine or string)
    :param response: The AI's response (can be a coroutine or string)
    """
    # Ensure query and response are strings
    if asyncio.iscoroutine(query):
        query = await query
    
    if asyncio.iscoroutine(response):
        response = await response
    
    # Convert to string and handle None
    query = str(query) if query is not None else ""
    
    # Convert response to JSON format
    try:
        # If response is already a dict/list, it will be properly serialized
        # If it's a string, wrap it in a dict
        if isinstance(response, str):
            response_json = {"text": response}
        else:
            response_json = response if response is not None else {}
        
        with get_db_cursor(commit=True) as cur:
            cur.execute("""
                INSERT INTO chat_history (session_id, query, response)
                VALUES (%s, %s, %s)
            """, (session_id, query, json.dumps(response_json)))
            
    except Exception as e:
        print(f"Error saving chat history: {e}")
        print(f"Debug info - session_id: {session_id}")
        print(f"Debug info - query: {query}")
        print(f"Debug info - response: {response}")
        print(f"Debug info - response_json: {response_json}")
        raise
# getting last 10 messages

async def get_chat_history(session_id):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT query, response, timestamp FROM chat_history WHERE session_id = %s ORDER BY timestamp DESC LIMIT 10", (session_id,))
        result = cursor.fetchall()
        history = []
        for row in cursor.fetchall():
            history.append({
            "query": row[0],
            "response": json.loads(row[1]),
            "timestamp": row[2].isoformat()
        })
        return history
    
    

# a=get_session_id("encoder_decoder.pdf")
# print(a)re
# if __name__ == "__main__":
#     try:
#         create_tables()
#         print("Tables created successfully.")

#     except Exception as e:
#         print(f"An error occurred: {e}")

