from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = Ollama(model='llama3:latest', request_timeout=30.0)

# Initialize the parser
parser = LlamaParse(result_type='markdown', api_key="LLAMA_CLOUD_API_KEY")

# Load documents using a file extractor
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("../data/Ample", file_extractor=file_extractor).load_data()

# Initialize the embedding model
embed_model = resolve_embed_model("local:BAAI/bge-m3")

# Create the vector index from the documents
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Initialize the query engine
query_engine = vector_index.as_query_engine(llm=llm)

# Function to reframe a user's question using the Query Engine
def reframe_question(user_question):
    # Define the prompt for reframing
    reframe_prompt = f"Reframe the following question while retaining its meaning: {user_question}"

    # Query the engine with the prompt
    response = query_engine.query(reframe_prompt)
    
    # Return the reframed question
    return response.response

# Example usage
user_question = "What is the avg Ph, and the standard Deviation Review of analytical data of the consignment received during the year?"
reframed_question = reframe_question(user_question)
print("Reframed Question:", reframed_question)
