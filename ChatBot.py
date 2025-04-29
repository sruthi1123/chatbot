import os
import tempfile
import streamlit as st
from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, vector_stores
from llama_index.core.embeddings import resolve_embed_model
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
#from llama_index.vector_stores import SimpleVectorStore
from dotenv import load_dotenv
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
 
 
# Load environment variables
load_dotenv()
 
# Streamlit application title
st.title("ChatBot")
 
# Function to initialize the LLM and document parser
def initialize_llm():
    return Ollama(model='llama3:latest', request_timeout=30.0)
 
def llama_parser():
    return LlamaParse(result_type='markdown', api_key = "LLAMA_CLOUD_API_KEY")
 
# Initialize LLM and Parser
ok_llm = initialize_llm()
parser = llama_parser()
 
# Create a temporary folder
def create_temp_folder():
    temp_dir = os.path.join(tempfile.gettempdir(), "streamlit_pdf_processing")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir
 
# Function to process uploaded PDF file
def process_pdf(uploaded_file, temp_dir):
    pdf_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(pdf_file_path, 'wb') as f:
        f.write(uploaded_file.read())
    return pdf_file_path
 
# Allow users to upload a PDF file
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
 
if uploaded_pdf:
    temp_folder = create_temp_folder()
    pdf_file_path = process_pdf(uploaded_pdf, temp_folder)
 
    # Load and parse the document
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(temp_folder, file_extractor=file_extractor).load_data()
 
    # Initialize the vector store index
    embed_model = resolve_embed_model("local:BAAI/bge-m3")
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
   
   
    # Build retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
        vector_store_query_mode="default"
    )
   
   
   
    # Build query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer = get_response_synthesizer()
    )
   
 
    st.success("PDF uploaded and processed successfully!")
 
    # User query input
    query = st.text_input("Enter your query:")
   
    if query:
        # Query the engine
        response = query_engine.query(query)
 
        if response:
            st.write("Relevant Information:")
            st.write(response)  # Display the response directly
        else:
            st.write("No relevant information found.")
else:
    st.warning("Please upload a PDF file to proceed.")
