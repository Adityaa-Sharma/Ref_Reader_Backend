import psycopg2
from psycopg2 import sql
import contextlib
from contextlib import contextmanager
import uuid

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
        

        
        # Create pdfs table with UUID foreign key
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pdfs (
                id SERIAL PRIMARY KEY,
                session_id UUID REFERENCES sessions(session_id),
                pdf_name VARCHAR NOT NULL,
                citations TEXT NOT NULL
            )
        """)
        ## creat table for paper , in this it will the , paper id (incremental),session id , paper name , authors , arxiv id(can be null)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id SERIAL PRIMARY KEY,
                session_id UUID REFERENCES sessions(session_id),
                paper_name VARCHAR NOT NULL,
                authors VARCHAR NOT NULL,
                arxiv_id VARCHAR
            )
        """)
                    


def save_pdf(session_id: str, pdf_name: str, citations: str):
    with get_db_cursor(commit=True) as cur:
        cur.execute("SELECT session_id FROM sessions WHERE session_id = %s", (session_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO sessions (session_id) VALUES (%s)", (session_id,))
        
        cur.execute("SELECT id FROM pdfs WHERE session_id = %s AND pdf_name = %s", (session_id, pdf_name))
        if not cur.fetchone():
            # Insert PDF entry
            cur.execute("""
                INSERT INTO pdfs (session_id, pdf_name, citations)
                VALUES (%s, %s, %s)
            """, (session_id, pdf_name, citations))


def get_pdf_citations(session_id):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT citations FROM pdfs WHERE session_id = %s", (session_id,))
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
 
    
    

# a=get_session_id("encoder_decoder.pdf")
# print(a)
if __name__ == "__main__":
    try:
        create_tables()
        print("Tables created successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")