import os
import tempfile
import streamlit as st
import torch
from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")  # Use environment variable for API key
EMBEDDING_MODEL = "local:BAAI/bge-m3"
LLM_MODEL = 'llama3:latest'
REQUEST_TIMEOUT = 60.0

def log_gpu_memory(stage):
    """Log the GPU memory usage at a specific stage."""
    allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
    print(f"{stage} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    st.write(f"{stage} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def initialize_llm():
    """Initialize the LLM and log GPU memory usage."""
    st.write("Initializing Llama3 model...")
    llm = Ollama(model=LLM_MODEL, request_timeout=REQUEST_TIMEOUT)
    log_gpu_memory("After loading Llama3 model")
    return llm

def create_temp_folder():
    """Create and clean a temporary folder for PDF processing."""
    temp_dir = os.path.join(tempfile.gettempdir(), "streamlit_pdf_processing")
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))  # Attempt to delete
            except PermissionError:
                st.warning(f"Could not delete {file}. It may still be in use.")
    else:
        os.makedirs(temp_dir)
    return temp_dir

def process_pdf(uploaded_file, temp_dir):
    """Save uploaded PDF to a temporary folder and return the file path."""
    pdf_file_path = os.path.join(temp_dir, uploaded_file.name)

    # Ensure the file is properly closed after writing
    with open(pdf_file_path, 'wb') as f:
        f.write(uploaded_file.read())
    
    return pdf_file_path

def load_documents_from_files(uploaded_files):
    """Load documents using SimpleDirectoryReader."""
    temp_folder = create_temp_folder()

    for uploaded_file in uploaded_files:
        process_pdf(uploaded_file, temp_folder)  # Save the file to temp_folder

    # Use SimpleDirectoryReader to load and parse documents
    file_extractor = {".pdf": LlamaParse(result_type='markdown', api_key=LLAMA_CLOUD_API_KEY)}
    documents = SimpleDirectoryReader(temp_folder, file_extractor=file_extractor).load_data()
    return documents

def create_vector_index(_documents):
    """Create a vector index from the documents and log GPU memory usage."""
    st.write("Resolving embedding model...")
    embed_model = resolve_embed_model(EMBEDDING_MODEL)
    log_gpu_memory("After resolving embedding model")
    
    st.write("Creating vector index...")
    vector_index = VectorStoreIndex.from_documents(_documents, embed_model=embed_model)
    log_gpu_memory("After creating vector index")
    return vector_index

def query_engine(vector_index, llm):
    """Create a query engine from the vector index."""
    return vector_index.as_query_engine(llm=llm)

def main():
    """Main function to run the Streamlit app."""
    st.title("DocQuery AI")

    # Initialize components
    llm = initialize_llm()

    # Upload files section
    uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
    if not uploaded_files:
        st.info("Please upload at least one PDF file to proceed.")
        return

    # Load documents from uploaded files
    #temp_folder = create_temp_folder()
    documents = load_documents_from_files(uploaded_files)
    if not documents:
        st.error("No valid documents loaded. Please check the uploaded files.")
        return

    # Create vector index and query engine
    vector_index = create_vector_index(_documents=documents)
    engine = query_engine(vector_index, llm)

    # Input query
    user_query = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        if user_query:
            try:
                results = engine.query(user_query)
                st.write("### Results:")
                st.write(results.response)  # Display the main response text
            except Exception as e:
                st.error(f"Error during query execution: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
