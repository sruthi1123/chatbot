import pymssql
import re
import csv
import os

# Database connection settings
def connect_to_db():
    """Connect to the SQL Server database using pymssql."""
    return pymssql.connect(
        host='ALtest05_dbs\SQL2019',
        user='sa',
        password='ALapqr@24',
        database='AL_DB_EAPQRAPQRDev'
    )

def get_procedure_definition(connection, sp_name):
    """Fetch the definition of the stored procedure."""
    try:
        query = """
        SELECT definition 
        FROM sys.sql_modules
        WHERE OBJECT_NAME(object_id) = %s;
        """
        with connection.cursor() as cursor:
            cursor.execute(query, (sp_name,))
            result = cursor.fetchone()
            if result:
                return result[0]  # The definition text
            else:
                print(f"Stored procedure {sp_name} not found.")
                return None
    except Exception as e:
        print(f"Error fetching definition for {sp_name}: {e}")
        return None

def extract_sub_procedures(procedure_definition):
    """Extract sub-stored procedures from the procedure definition."""
    try:
        # Use regex to find EXEC/EXECUTE statements
        matches = re.findall(r'\bEXEC(?:UTE)?\s+([\w\[\]]+)', procedure_definition, re.IGNORECASE)
        return list(set(matches))  # Return unique matches
    except Exception as e:
        print(f"Error parsing procedure definition: {e}")
        return []

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

def main():
    """Main function to execute the main stored procedure and fetch all resultsets."""
    connection = connect_to_db()

    # Main stored procedure name
    main_sp_name = "AL_SP_APQRFormulation_Report_Category"

    # Hardcoded parameter values for the main stored procedure
    param_values = ["191224", "PANTO", "01/01/2024", "31/12/2024"]

    # Execute the main stored procedure
    print(f"Executing main stored procedure: {main_sp_name}...")
    all_results = execute_stored_procedure(connection, main_sp_name, param_values)

    # Save all resultsets to CSV files
    if all_results:
        save_results_to_csv(all_results, folder_name="results_folder", base_filename="main_sp_results")
        print("All resultsets saved to CSV in the 'results_folder'.")
    else:
        print("No data returned by the main stored procedure.")

    connection.close()

if __name__ == "__main__":
    main()
