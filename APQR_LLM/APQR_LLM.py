from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
# from prompts import context

from CPP_KPP import CPP_Calculation, KPP_Calculation, standard_deviation

from dotenv import load_dotenv

load_dotenv()

llm = Ollama(model = 'llama3:latest', request_timeout = 30.0)

parser = LlamaParse(result_type = 'markdown', api_key = "LLAMA_CLOUD_API_KEY")

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("../data/Ample", file_extractor = file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model = embed_model)
query_engine = vector_index.as_query_engine(llm = llm)



# tools = [CPP_Calculation, KPP_Calculation, standard_deviation]
# agent = ReActAgent.from_tools(tools, llm = llm, verbose = True, context = context)

## Working
# results = query_engine.query("What is the PH Limits of Review of analytical data of the consignment received during the year?")
# results = query_engine.query("What is the avg Ph, and the standard Deviation Review of analytical data of the consignment received during the year?")
# results = query_engine.query("What is the name of the vendor of sodium hydroxide?"
# results = query_engine.query("For batch B000123003 in the Review of Filtration, Filling, Lyophilization and Sealing process, give me the results for all the parameters?")
# results = query_engine.query("What is the Specification Limits for Review of Diluent COA?")
# results = query_engine.query("Give me the details for the Product Information?")
# results = query_engine.query("In the Review of Diluent COA, give me the batch that defies the Specification limit which is Pure Blue colour for the parameter calcium and magnesium?")
# results = query_engine.query("What is the avg Ph, and the standard Deviation Review of analytical data of the consignment received during the year?")

# query = "What are the average PH and standard deviation values for the Parameters section in the Review of analytical data of the consignment received during the year?"
# query = "Which country has a dossier submission date of November 3rd, 2018, according to the context information provided?"
# query = "What is the standard deviation of the values for the Parameters section in the Review of analytical data of the consignment received during the year?"
# query = "In Temperature Relative Humidity Results monitored, Give me all the values counts?"
# query = "Review of WFI, Purified Water, Pure Steam, Soft Water Montering?"
# query = "In the Review of Non-Viable Monitoring, Give me the values count of the Grades?"
# query = "In the Review of process loss, What is the avg and standard Deviation of the Rubber bungs?"
# query = "Does Amplelogic provide QMS software?"
# query = "In the Review of Dossier Registration And Approvals and verifications / Changes made in the registered Dossiers, How Many Have the Status as Registered?"
# query = "What are Mfg. Date in Data of the total number of batches manufactured,released and rejected during the review period"
# query = "In the Washing & Depyrogenation details, Calculate the Standard Deviation for recycle water pressure (Max)?"  

# query = "What is the Limit for the Nitrogen Gas Monitering?"
# query = "In the Review of Technical agreements, Which Tracking No, got it's approval on 01/02/2019?"
# query = "In the Washing & Depyrogenation details, What are all the Specification Limits?"

# query = "For the Specifications of :Sodium Hydroxide, What is the average PH?"

# query = "What is the Manufacturing Yield, for the Batch no B000123002?"
query = "Give me all the details of Review of details for Batch Release/ Rejection/ Recall during the Year?"

# query = "In Review of master batch manufacturing records(BMR),master packaging records(BPR) and master manufacturing formula(MMF), what are all the Document Names?"
# Perform the query
response = query_engine.query(query)


from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate

# try chaining basic prompts
prompt_str = """ Given a response {response_str}, check whether the response is correct according to the User's query {query}.
If the response is correct say Yes, else say No.
It's a no if reponse does not provide sufficient information.
I just need Answer as Yes or No. """

prompt_tmpl = PromptTemplate(prompt_str)

p = QueryPipeline(chain = [prompt_tmpl, llm], verbose = True)
output = p.run(response_str = response, query = query)
print(output)

if "no" in str(output).lower():
    prompt_str_2 = """Given a query, I want you to find the main Heading from the query {query}."""
    prompt_tmpl_2 = PromptTemplate(prompt_str_2)
    
    p_2 = QueryPipeline(chain = [prompt_tmpl_2, llm], verbose = True)
    main_heading  = p_2.run(query = query)
    print(main_heading)

    response_2 = query_engine.query("Give me the content related to " + str(main_heading))
    # Print the answer
    print(f"Here is the content related to {main_heading}:\n response")
    
else :
    # Print the answer
    print("Answer:\n", response)