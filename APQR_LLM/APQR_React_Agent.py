from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from prompts import context

from CPP_KPP import cpp_tool, kpp_tool, std_dev_tool

from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = Ollama(model = 'llama3:latest', request_timeout = 30.0)

# Set up the parser for documents
parser = LlamaParse(result_type = 'markdown', api_key = "LLAMA_CLOUD_API_KEY")

# Load documents from the specified directory
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data/Ample", file_extractor=file_extractor).load_data()

# Resolve embedding model
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm = llm)

# Set up tools for the agent
tools = [
    QueryEngineTool(
        query_engine = query_engine,
        metadata = ToolMetadata(
            name = "Chat_with_Pdfs",
            description = "Hello World")
    ),
    cpp_tool, kpp_tool, std_dev_tool
]

# Create the ReAct agent
agent = ReActAgent.from_tools(tools, llm = llm, verbose = True, context = context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
    
# # Example query to the agent
# results = agent.query("Review of WFI, Purified Water, Pure Steam, Soft Water Monitoring?")
# print(results)
