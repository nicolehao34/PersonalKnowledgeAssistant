from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
from PersonalKnowledgeAssistant.document_processor import DocumentProcessor
from PersonalKnowledgeAssistant.rag_engine import RAGEngine

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    # Save the uploaded file
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # TODO: include different document types, pdf, docx, etc.
        text = DocumentProcessor.process_document(file_path)
        
        # Add to RAG engine
        rag_engine.add_documents(
            texts=[text],
            metadata=[{"filename": file.filename}]
        )
        
        return {"message": "Document processed successfully"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up the uploaded file
        file_path.unlink(missing_ok=True)

@app.post("/query")
async def query_documents(question: str = Form(...)):
    """Query the RAG system."""
    try:
        response = rag_engine.query(question)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 