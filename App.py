import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from PyPDF2 import PdfReader

# Load environment variables from .env file
load_dotenv()

# Constants
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")  # Use environment variable for API key
print(f"LLAMA_CLOUD_API_KEY: {LLAMA_CLOUD_API_KEY}")
EMBEDDING_MODEL = "local:BAAI/bge-m3"
LLM_MODEL = 'llama3:latest'
REQUEST_TIMEOUT = 30.0
PRE_UPLOADED_PDF_PATH = r"C:\Users\aiml.ALTESTSERVER02\ChatBot\Streamlit\PDF_Folder"  # Path to the pre-uploaded PDF

def initialize_llm():
    """Initialize the LLM."""
    return Ollama(model=LLM_MODEL, request_timeout=REQUEST_TIMEOUT)

def initialize_parser():
    """Initialize the document parser."""
    return LlamaParse(result_type='markdown', api_key=LLAMA_CLOUD_API_KEY)

def load_documents(file_path, file_extractor):
    """Load documents from the specified file path."""
    reader = SimpleDirectoryReader(file_path, file_extractor=file_extractor)
    return reader.load_data()

def create_vector_index(documents):
    """Create a vector index from the documents."""
    embed_model = resolve_embed_model(EMBEDDING_MODEL)
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

def query_engine(vector_index, llm):
    """Create a query engine from the vector index."""
    return vector_index.as_query_engine(llm=llm)

def main():
    """Main function to run the Streamlit app."""
    st.title("Document Query System")
    
    # Initialize components
    llm = initialize_llm()
    parser = initialize_parser()
    file_extractor = {".pdf": parser}

    # Load documents from the pre-uploaded PDF
    documents = load_documents(PRE_UPLOADED_PDF_PATH, file_extractor)
    if not documents:
        st.error("No documents loaded. Please check the PDF file path.")
        return

    # Create vector index and query engine
    vector_index = create_vector_index(documents)
    engine = query_engine(vector_index, llm)

    # Input query
    user_query = st.text_input("Enter your query:")
    
    if st.button("Submit Query"):
        if user_query:
            try:
                results = engine.query(user_query)
                st.write("### Results:")
                st.write(results.response)    # To  Display only the main response text

# Display source details if available
                #if results.source_nodes:
                    #st.write("### Sources:")
                    #for source_node in results.source_nodes:
                        #metadata = source_node.node.metadata
                        #file_path = metadata.get('file_path', 'N/A')
                        #file_name = metadata.get('file_name', 'N/A')
                        #st.write(f"- **File Name**: {file_name}")
                        #st.write(f"  **File Path**: {file_path}")
#Display source details if available                    
            except Exception as e:
                st.error(f"Error during query execution: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
