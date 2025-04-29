import pymssql

# Database connection settings
def connect_to_db():
    """Connect to the SQL Server database using pymssql."""
    return pymssql.connect(
        host='ALtest05_dbs\\SQL2019',
        user='sa',
        password='ALapqr@24',
        database='AL_DB_EAPQRAPQRDev',
        autocommit=True  # Enable autocommit for session-level operations
    )

def execute_stored_procedure(connection, sp_name, params):
    """Execute a stored procedure and log SQL Server messages."""
    try:
        sql = f"EXEC {sp_name} " + ", ".join(f"'{param}'" if param else "NULL" for param in params)
        print(f"Executing SQL: {sql}")
        with connection.cursor() as cursor:
            cursor.execute(sql)

            # Capture and log SQL Server messages
            while cursor.messages:
                message = cursor.messages.pop(0)
                print(f"SQL Server Message: {message[0]}")

            # Check if a resultset is available
            results = []
            while True:
                if cursor.description:  # Resultset is available
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    results.append([dict(zip(columns, row)) for row in rows])
                if not cursor.nextset():  # Move to the next resultset
                    break
            return results
    except pymssql.OperationalError as e:
        print(f"Operational Error: {e}")
    except pymssql.ProgrammingError as e:
        print(f"Programming Error: {e}")
    except Exception as e:
        print(f"Error executing stored procedure {sp_name}: {e}")
    return []

def main():
    """Main function to test a single stored procedure."""
    # Connect to the database
    connection = connect_to_db()

    # Stored procedure name
    sp_name = "AL_SP_APQRFormulation_Report_Category"

    # Hardcoded parameter values
    param_values = ["1410", "LEVO_API", "01/01/2024", "30/12/2024"]

    # Execute the stored procedure
    print("Starting stored procedure execution...")
    results = execute_stored_procedure(connection, sp_name, param_values)

    # Display the results
    if results:
        print("Results:")
        for resultset in results:
            for row in resultset:
                print(row)
    else:
        print("No data returned. Check the stored procedure logic or input parameters.")

    # Close the connection
    connection.close()

if __name__ == "__main__":
    main()
