# Advanced RAG With LlamaParse

import nest_asyncio
nest_asyncio.apply()


from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context

# Setting up the LLM with Ollama
llm = Ollama(model = 'llama3:latest', request_timeout = 30.0)
embed_model = resolve_embed_model("local:BAAI/bge-m3")

Settings.llm = llm
Settings.embed_model = embed_model

# Using `LlamaParse` PDF reader for PDF Parsing
# We also compare two different retrieval/query engine strategies:
# 1. Using raw Markdown text as nodes for building index and apply simple query engine for generating the results;
# 2. Using MarkdownElementNodeParser for parsing the LlamaParse output Markdown results and building recursive retriever query engine for generation.

documents = LlamaParse(result_type = "markdown").load_data("./data/Ample/Amplelogic_APQR_testing.pdf")

from copy import deepcopy
from llama_index.core.schema import TextNode

# Splitting the document into page nodes
def get_page_nodes(docs, separator = "\n---\n"):
    """ Split each document into page node, by separator."""
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split(separator)
        for doc_chunk in doc_chunks:
            node = TextNode(
                text = doc_chunk,
                metadata = deepcopy(doc.metadata),
            )
            nodes.append(node)
    return nodes

page_nodes = get_page_nodes(documents)

from llama_index.core.node_parser import MarkdownElementNodeParser
node_parser = MarkdownElementNodeParser(
    llm = llm, num_workers = 8
)

# Parsing the document
nodes = node_parser.get_nodes_from_documents(documents)
base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

# Create a recursive vector store index
recursive_index = VectorStoreIndex(nodes = base_nodes + objects + page_nodes)

from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)

# Setting up the recursive query engine
recursive_query_engine = recursive_index.as_query_engine(
    similarity_top_k = 5, node_postprocessors = [reranker], verbose = True
)

print(len(nodes))

# # Setup Baseline
# # For comparison, we setup a naive RAG pipeline with default parsing and standard chunking, indexing, retrieval.
# reader = SimpleDirectoryReader(input_files = ["./data/Ample/Amplelogic_APQR_testing.pdf"])
# base_docs = reader.load_data()
# raw_index = VectorStoreIndex.from_documents(base_docs)
# raw_query_engine = raw_index.as_query_engine(
#     similarity_top_k = 5, node_postprocessors = [reranker]
# )

# # Using LlamaParser as PDF data parsing methods and retrieve tables with two different methods
# # We compare base query engine vs recursive query engine with tables

# # Table Query Task: Queries for Table Question Answering
# query = "In the Washing & Depyrogenation details, What are all the Specification Limits?"

# response_1 = raw_query_engine.query(query)
# print("\n***********Basic Query Engine***********")
# print(response_1)

# response_2 = recursive_query_engine.query(query)
# print("\n***********LlamaParse + Recursive Retriever Query Engine***********")
# print(response_2)
