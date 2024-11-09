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

def load_session_history(session_id: str):
    messages = []
    with get_db_cursor() as cur:
        cur.execute("""
            SELECT role, content
            FROM messages
            WHERE session_id = %s
            ORDER BY id
        """, (session_id,))
        for role, content in cur.fetchall():
            messages.append({"role": "user" if role == "Human" else "assistant", "content": content})
    return messages


def get_pdf_citations(pdf_name):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT session_id, citations FROM pdfs WHERE pdf_name = %s", (pdf_name,))
        result = cursor.fetchone()
        if result:
            return {"session_id": result[0], "citations": result[1]}
        return None


# if __name__ == "__main__":
#     try:
#         create_tables()
#         print("Tables created successfully.")

#     except Exception as e:
#         print(f"An error occurred: {e}")