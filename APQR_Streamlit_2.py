import os
import shutil
import streamlit as st
from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.query_pipeline import QueryPipeline
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Streamlit application title
st.title("ChatBot")

# Function to initialize the LLM and document parser
def initialize_llm():
    llm = Ollama(model = 'llama3:latest', request_timeout=30.0)
    return llm

def llama_parser():
    parser = LlamaParse(result_type='markdown', api_key="LLAMA_CLOUD_API_KEY")
    return parser

# Initialize LLM and Parser
llm = initialize_llm()
parser = llama_parser()

# Create a temporary folder if it doesn't exist
def create_temp_folder():
    temp_dir = os.path.join(tempfile.gettempdir(), "streamlit_pdf_processing")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir


# Function to process uploaded PDF file
def process_pdf(uploaded_file, temp_dir):
    # Save the uploaded PDF in the temp folder
    pdf_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(pdf_file_path, 'wb') as f:
        f.write(uploaded_file.read())
    return pdf_file_path

# Allow users to upload a PDF file
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    # Create (or reuse) the temp folder and process the uploaded PDF
    temp_folder = create_temp_folder()
    pdf_file_path = process_pdf(uploaded_pdf, temp_folder)

    # Load and parse the document using SimpleDirectoryReader
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(temp_folder, file_extractor=file_extractor).load_data()

    # Embed model and Vector Store Index initialization
    embed_model = resolve_embed_model("local:BAAI/bge-m3")
    vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    
    # File-level retriever
    doc_retriever = vector_index.as_retriever(
        retrieval_mode="files_via_content",
        files_top_k=1
    )

    # Chunk-level retriever
    chunk_retriever = vector_index.as_retriever(
        retrieval_mode="chunks",
        rerank_top_n=5
    )
    
    # Example user question
    user_question = "What is the procedure for equipment cleaning in the Production department?"

    # Retrieve relevant documents (file-level retrieval)
    relevant_documents = doc_retriever.retrieve(user_question)

    # Retrieve relevant chunks (chunk-level retrieval)
    relevant_chunks = chunk_retriever.retrieve(user_question)

    # Print out the results
    print("Relevant Documents:")
    for doc in relevant_documents:
        print(doc)

    print("\nRelevant Chunks:")
    for chunk in relevant_chunks:
        print(chunk)

    
    query_engine = vector_index.as_query_engine(llm=llm)

    st.success("PDF uploaded and processed successfully!")

    # Enter the query
    query = st.text_input("Enter your query:", placeholder="Ask something about the document...")

    if st.button("Submit Query"):
        with st.spinner("Fetching response..."):
            # Query the engine
            response = query_engine.query(query)

            # Query validation pipeline
            prompt_str = """Given a response {response_str}, check whether the response is correct according to the User's query {query}.
            If the response is correct, say Yes, else say No. I just need Answer as Yes or No."""
            prompt_tmpl = PromptTemplate(prompt_str)
            p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)
            output = p.run(response_str=response, query=query)

            if "no" in str(output).lower():
                # Retry the query if validation fails
                response = query_engine.query(query)

            st.markdown(f"**Response:**\n\n{response}")
else:
    st.warning("Please upload a PDF file to proceed.")