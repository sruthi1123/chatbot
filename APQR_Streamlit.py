import os
import streamlit as st
from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.query_pipeline import QueryPipeline
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Streamlit application title
st.title("APQR PDF ChatBot")

# Function to initialize the LLM and document parser
def initialize_llm():
    llm = Ollama(model = 'llama3:latest', request_timeout=30.0)
    return llm

def llama_parser():
    parser = LlamaParse(result_type='markdown', api_key = "LLAMA_CLOUD_API_KEY")
    return parser

# Initialize LLM and Parser
llm = initialize_llm()
parser = llama_parser()

# Predefined file location
pdf_file_path = r"C:\Users\aiml.ALTESTSERVER02\ChatBot\Streamlit\PDF_Folder\Amplelogic_APQR_testing1.pdf"
# Load and parse the document using SimpleDirectoryReader
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(r"C:\Users\aiml.ALTESTSERVER02\ChatBot\Streamlit\PDF_Folder", file_extractor=file_extractor).load_data()

# Embed model and Vector Store Index initialization
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

st.success("Predefined document loaded successfully!")

# Check if the PDF exists
if os.path.exists(pdf_file_path):
    # Enter the query
    query = st.text_input("Enter your query:", placeholder = "Ask something about the document...")

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

            # st.write(f"Query validation result: {output}")

            if "no" in str(output).lower():
                # Query the engine
                response = query_engine.query(query)

                # Query validation pipeline
                prompt_str = """Given a response {response_str}, check whether the response is correct according to the User's query {query}.
                If the response is correct, say Yes, else say No. I just need Answer as Yes or No."""
                prompt_tmpl = PromptTemplate(prompt_str)
                p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)
                output = p.run(response_str=response, query=query)
                
            st.markdown(f"**Response:**\n\n{response}")
else:
    st.error(f"PDF file '{pdf_file_path}' not found. Please check the path.")
