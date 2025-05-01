from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import shutil

DATA_PATH = "data/books"

def main():
    # Load documents from the specified directory
    documents = load_documents()
    
    # Chunk the documents into smaller pieces
    chunks = chunk_documents(documents)
    
    # Save the chunks to a Chroma database
    save_to_chroma(chunks)
    
    
# Load documents
# The DirectoryLoader is a class that loads documents from a specified directory
# The glob parameter specifies the pattern for the files to be loaded
# In this case, it loads all markdown files in the data/books directory
def load_documents():
    """Load documents from the specified directory."""
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md")
    documents = loader.load()
    return documents

# text chunking/splitting

def chunk_documents(documents):
    """Chunk documents into smaller pieces."""
    # Initialize the text splitter
    # The RecursiveCharacterTextSplitter is a text splitter that splits text into chunks based on character count
    # It allows for overlapping chunks, which can be useful for certain applications
    # The chunk size and overlap can be adjusted based on the specific needs of the application
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # a chunk size of 1000 characters
        chunk_overlap=200, # an overlap of 200 characters
        length_function=len,
        add_start_index = True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    document = chunks[10]
    
    print(document.page_content)
    print(document.metadata) # source, start index
    
    return chunks

def save_to_chroma(chunks):
    CHROMA_PATH = "chroma"
    
    
    # create a new db from the documents
    # OpenAIEmbeddings is a class that provides embeddings for text using OpenAI's API
    # persist_directionary is the directory where the database will be stored
    
    # Clear out database first, if it exists
    if os.path.exists(CHROMA_PATH):
        print(f"Deleting existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    # Create a new Chroma database from the chunks
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    
    # force to save using the persist method
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")
    
    
# prepare the db
embedding_function = OpenAIEmbeddings()

db = Chroma()
    
if __name__ == "__main__":
    main()