from typing import List, Union
import PyPDF2
from docx import Document
from pathlib import Path

class DocumentProcessor:
    """Handles processing of PDF and Word documents."""
    
    @staticmethod
    def process_pdf(file_path: Union[str, Path]) -> str:
        """Extract text from a PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    @staticmethod
    def process_docx(file_path: Union[str, Path]) -> str:
        """Extract text from a Word document."""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    @staticmethod
    def process_document(file_path: Union[str, Path]) -> str:
        """Process a document based on its file extension."""
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.pdf':
            return DocumentProcessor.process_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return DocumentProcessor.process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}") 