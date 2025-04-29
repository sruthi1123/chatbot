import streamlit as st
from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.query_pipeline import QueryPipeline
from dotenv import load_dotenv
import requests  # For web search

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = Ollama(model='llama3:latest', request_timeout=30.0)

# Initialize the parser
parser = LlamaParse(result_type = 'markdown', api_key = "LLAMA_CLOUD_API_KEY")

# Load documents using the parser
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(r"C:\Users\aiml.ALTESTSERVER02\ChatBot\Streamlit\PDF_Folder", file_extractor=file_extractor).load_data()

# Resolve the embedding model
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

# Streamlit UI
st.title("APQR ChatBot")
st.write("Ask analytical data-related queries.")

# Query input from the user
query = st.text_area("Enter your query", "In the Washing & Depyrogenation details, What are all the Specification Limits?")

# Function to perform a basic web search using SerpAPI
def basic_web_search(query):
    api_key = 'SERP_API_KEY'  
    search_url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "api_key": api_key
    }
    
    try:
        response = requests.get(search_url, params = params)
        results = response.json()
        search_results = []
        if 'organic_results' in results:
            for result in results['organic_results']:
                search_results.append({
                    'title': result['title'],
                    'link': result['link'],
                    'snippet': result.get('snippet', '')
                })
        return search_results
    except Exception as e:
        st.error(f"An error occurred during the web search: {e}")
        return None

# Perform the query when the "Run Query" button is clicked
if st.button("Run Query"):
    response = query_engine.query(query)
    
    # Extract the raw response text
    response_text = response.response if hasattr(response, 'response') else str(response)
    
    # st.write("Response:", response_text)

    # Chaining prompt to check yes/no validation
    prompt_str = """ Given a response {response_str}, check whether the response is Yes or No. 
    If you find any negative response, such as no such results or couldn't find any answer, reponse by saying No.
    I just need Answer as Yes or No based on the response. """
    prompt_tmpl = PromptTemplate(prompt_str)
    p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)
    output = p.run(response_str=response_text)

    print(f"Prompt Output: {output}")
    
    if "no" in str(output).lower():
        response_2 = query_engine.query(query)
    
        # Extract the raw response text
        response_text_2 = response_2.response if hasattr(response_2, 'response') else str(response_2)
        
        st.write("Response:", response_text_2)
        
    else:
        st.write("Answer:", response_text)

# Perform a web search when the "Web Search" button is clicked
if st.button("Web Search"):
    search_results = basic_web_search(query)
    if search_results:
        st.write("Web Search Results:")
        for i, result in enumerate(search_results):
            st.write(f"{i+1}. [{result['title']}]({result['link']})\n{result['snippet']}")
    else:
        st.write("No relevant web results found.")