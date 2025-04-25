from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the RAG engine with vector store and LLM."""
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the vector store."""
        if metadata is None:
            metadata = [{} for _ in texts]
            
        # Split texts into chunks
        documents = []
        for text, meta in zip(texts, metadata):
            chunks = self.text_splitter.split_text(text)
            documents.extend([(chunk, meta) for chunk in chunks])
            
        # Add to vector store
        self.vectorstore.add_texts(
            texts=[doc[0] for doc in documents],
            metadatas=[doc[1] for doc in documents]
        )
        self.vectorstore.persist()
        
    def query(self, question: str, k: int = 4) -> str:
        """Query the RAG system with a question."""
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k})
        )
        
        response = qa_chain.run(question)
        return response 