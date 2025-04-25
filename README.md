# Personal Knowledge Assistant

A RAG-based document processing and querying system that can handle PDF and Word documents.

## Features

- Document processing for PDF and Word documents
- RAG-based question answering using GPT-3.5-turbo
- FastAPI backend for document upload and querying
- Vector storage using ChromaDB
- Sentence transformers for embeddings

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the server:
```bash
python app.py
```

2. The API will be available at `http://localhost:8000`

3. API Endpoints:
   - POST `/upload`: Upload a document (PDF or Word)
   - POST `/query`: Ask questions about your documents

## Example Usage

### Upload a document
```bash
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/upload
```

### Query the system
```bash
curl -X POST -F "question=What is the main topic of the document?" http://localhost:8000/query
```

## Architecture

- `document_processor.py`: Handles document processing for PDF and Word files
- `rag_engine.py`: Implements the RAG system with vector storage and LLM
- `app.py`: FastAPI application for document upload and querying

## Notes

- Documents are processed in chunks for better retrieval
- The system uses the all-MiniLM-L6-v2 model for embeddings
- Vector storage is persisted in the `chroma_db` directory
- Uploaded files are temporarily stored in the `uploads` directory 