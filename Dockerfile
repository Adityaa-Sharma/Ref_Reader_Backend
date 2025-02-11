FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up NLTK data directory and download required data
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p ${NLTK_DATA} && \
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('tokenizers/punkt', quiet=True)"

# Copy application code
COPY . .

# Copy .env into the container
COPY .env .env

# Make sure .env is readable by the app
RUN chmod 600 .env

# Create a directory for environment files
RUN mkdir -p /app/config

# Set permissions
RUN chmod -R 755 /app

# Environment variables will be passed during runtime
ENV AZURE_API_BASE=""
ENV AZURE_API_KEY=""
ENV AZURE_OPENAI_DEPLOYMENT_NAME=""
ENV AZURE_API_VERSION=""

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with environment variable check
CMD ["sh", "-c", "python -c 'from dotenv import load_dotenv; load_dotenv(); from services.Agents import get_llm_config; get_llm_config()' && uvicorn main:app --host 0.0.0.0 --port 8000"]
