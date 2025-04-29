import os
import tempfile
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv
import pymssql
import csv

# Load environment variables from .env file
load_dotenv()

# Constants
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")  # Use environment variable for API key
EMBEDDING_MODEL = "local:BAAI/bge-m3"
LLM_MODEL = 'llama3:latest'
REQUEST_TIMEOUT = 30.0

# Database connection settings
def connect_to_db():
    """Connect to the SQL Server database using pymssql."""
    return pymssql.connect(
        host='ALtest05_dbs\SQL2019',
        user='sa',
        password='ALapqr@24',
        database='AL_DB_EAPQRAPQRDev'
    )

def execute_stored_procedure(connection, sp_name, params):
    """Execute a stored procedure with parameters and fetch all resultsets."""
    try:
        sql = f"EXEC {sp_name} " + ", ".join(f"'{param}'" if param else "NULL" for param in params)
        print(f"Executing {sp_name}...")
        with connection.cursor() as cursor:
            cursor.execute(sql)
            
            all_results = []  # To store all resultsets
            while True:
                if cursor.description:  # Check if a resultset is available
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    resultset = [dict(zip(columns, row)) for row in rows]
                    all_results.append(resultset)  # Append the resultset
                if not cursor.nextset():  # Move to the next resultset
                    break
            return all_results
    except Exception as e:
        print(f"Error executing stored procedure {sp_name}: {e}")
        return []

def save_results_to_csv(all_results, folder_name="results_folder", base_filename="results"):
    """Save all resultsets to separate CSV files inside a specified folder."""
    # Create folder if it does not exist
    os.makedirs(folder_name, exist_ok=True)

    for idx, resultset in enumerate(all_results):
        if resultset:  # Only save non-empty resultsets
            filename = os.path.join(folder_name, f"{base_filename}_resultset_{idx + 1}.csv")
            with open(filename, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=resultset[0].keys())
                writer.writeheader()
                writer.writerows(resultset)
            print(f"Saved resultset {idx + 1} to {filename}")

def create_vector_index_from_csv(folder_name="results_folder"):
    """Manually create a vector index from CSV files using plain text embeddings."""
    from llama_index.core.vector_stores import SimpleVectorStore

    embed_model = resolve_embed_model(EMBEDDING_MODEL)
    vector_store = SimpleVectorStore()

    for file_name in os.listdir(folder_name):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_name, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                # Generate embeddings for the content
                embedding = embed_model.get_text_embedding(content)

                # Add the embedding and metadata to the vector store
                vector_store.add_embedding(
                    embedding=embedding,
                    metadata={"source": file_name},
                    text=content
                )

    # Build a VectorStoreIndex from the populated vector store
    return VectorStoreIndex(vector_store=vector_store)



def main():
    """Main function to run the Streamlit app."""
    st.title("DocQuery AI")

    try:
        # Database connection
        connection = connect_to_db()

        # Main stored procedure name
        main_sp_name = "AL_SP_APQRFormulation_Report_Category"

        # Hardcoded parameter values for the main stored procedure
        param_values = ["1410", "LEVO_API", "01/01/2024", "30/12/2024"]

        # Execute the main stored procedure
        st.write(f"Executing stored procedure: {main_sp_name}...")
        all_results = execute_stored_procedure(connection, main_sp_name, param_values)

        # Save all resultsets to CSV files
        if all_results:
            save_results_to_csv(all_results, folder_name="results_folder", base_filename="main_sp_results")
            st.success("Data saved to CSV files in the 'results_folder'.")
        else:
            st.error("No data returned by the stored procedure.")
            return

        # Create vector index from CSV files
        st.write("Creating vector index from CSV files...")
        vector_index = create_vector_index_from_csv(folder_name="results_folder")
        llm = initialize_llm()
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
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()

