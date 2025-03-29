import pyodbc # type: ignore
import os
from dotenv import load_dotenv

class Database:
    def __init__(self, connection_string):
        """
        Initializes the database connection.

        Parameters:
        - connection_string (str): SQL Server connection string.
        """
        self.connection_string = connection_string
        self.conn = None

    def connect(self):
        """Establishes a database connection."""
        try:
            # print(self.connection_string)
            self.conn = pyodbc.connect(self.connection_string)
            print("✅ Connected to Azure SQL Database successfully!")
        except pyodbc.Error as e:
            print(f"❌ Database connection failed: {e}")

    def create_table(self, table_name, schema):
        """
        Creates a table if it does not exist.

        Parameters:
        - table_name (str): The name of the table to create.
        - schema (dict): Dictionary where keys are column names and values are SQL data types.

        Example:
        ```python
        schema = {
            "id": "INT IDENTITY(1,1) PRIMARY KEY",
            "title": "NVARCHAR(500)",
            "content": "NVARCHAR(MAX)",
            "publish_date": "DATETIME",
            "authors": "NVARCHAR(500)",
            "source": "NVARCHAR(255)",
            "url": "NVARCHAR(1000) UNIQUE"
        }
        ```
        """
        if not self.conn:
            print("⚠️ No database connection.")
            return

        try:
            cursor = self.conn.cursor()

            # Generate column definitions dynamically
            columns = ", ".join([f"{col} {dtype}" for col, dtype in schema.items()])

            # SQL query to create table if it doesn't exist
            query = f"""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}')
                BEGIN
                    CREATE TABLE {table_name} ({columns});
                END
            """

            # Execute and commit
            cursor.execute(query)
            self.conn.commit()
            cursor.close()
            print(f"✅ Table '{table_name}' is ready.")

        except pyodbc.Error as e:
            print(f"❌ Error creating table '{table_name}': {e}")

    def check_table_exists(self, schema_name, table_name):
        """
        Checks if a table exists in an Azure SQL Database.

        :param server: Azure SQL Server name (e.g., 'your_server.database.windows.net')
        :param database: Database name
        :param username: SQL username
        :param password: SQL password
        :param schema_name: Schema name (e.g., 'dbo')
        :param table_name: Table name to check
        :return: True if the table exists, False otherwise
        """
        try:
            # Connect to Azure SQL Database
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()

            # SQL Query to check if table exists
            query = """
            SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """
            
            cursor.execute(query, (schema_name, table_name))
            table_exists = cursor.fetchone() is not None  # Check if any row is returned

            # Close connection
            cursor.close()
            conn.close()

            return table_exists
        except Exception as e:
            print(f"❌ Error checking table existence: {e}")
            return False  # Assume table does not exist if an error occurs

    def insert_record(self, table_name, record):
        """
        Inserts a record into the specified database table.

        Parameters:
        - table_name (str): The name of the table to insert into.
        - record (dict): A dictionary where keys are column names and values are data.
        """
        if not self.conn:
            print("⚠️ No database connection.")
            return

        if not record:
            print("⚠️ Empty record. Nothing to insert.")
            return
        
        try:
            cursor = self.conn.cursor()

            # Extract columns and values dynamically
            columns = ", ".join(record.keys())  # Convert keys to column names
            placeholders = ", ".join(["?" for _ in record])  # Create placeholders for SQL query
            values = tuple(record.values())  # Convert dictionary values to tuple

            # SQL Insert Query
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            
            # Execute and commit
            cursor.execute(query, values)
            self.conn.commit()
            print(f"✅ Record inserted successfully into '{table_name}'.")
            cursor.close()

        except pyodbc.Error as e:
            print(f"❌ Database insert failed: {e}")

    def insert_records_bulk(self, table_name, records):
        """
        Inserts multiple records into a table efficiently using a bulk insert.

        Parameters:
        - table_name (str): The name of the table.
        - records (list of dicts): A list of dictionaries where keys are column names, and values are data.
        """
        if not self.conn:
            print("⚠️ No database connection.")
            return

        if not records:
            print("⚠️ No records provided for bulk insert.")
            return

        try:
            cursor = self.conn.cursor()

            # Extract column names from the first record
            columns = ", ".join(records[0].keys())
            placeholders = ", ".join(["?" for _ in records[0]])

            # Prepare values as a list of tuples
            values = [tuple(record.values()) for record in records]

            # SQL Insert Query
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

            # Execute many records at once
            cursor.executemany(query, values)
            self.conn.commit()
            cursor.close()
            print(f"✅ {len(records)} records inserted successfully into '{table_name}'.")
        except pyodbc.Error as e:
            print(f"❌ Bulk insert failed: {e}")

    def execute_query(self, query, params=None, fetch_one=False, fetch_all=False, commit=False):
        """
        Execute any SQL query with optional parameters.

        Parameters:
        - query (str): The SQL query string.
        - params (tuple/list): Parameters for the query (optional).
        - fetch_one (bool): Whether to fetch one result (for SELECT).
        - fetch_all (bool): Whether to fetch all results (for SELECT).
        - commit (bool): Whether to commit the transaction (for INSERT/UPDATE/DELETE).

        Returns:
        - Query result(s) if fetch_one or fetch_all is True, else None.
        """
        if not self.conn:
            print("⚠️ No database connection.")
            return None

        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            result = None
            if fetch_one:
                result = cursor.fetchone()
            elif fetch_all:
                result = cursor.fetchall()

            if commit:
                self.conn.commit()

            cursor.close()
            return result

        except pyodbc.Error as e:
            print(f"❌ Error executing query: {e}")
            return None

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("🔌 Connection closed.")
    


# # Example usage
# if __name__ == "__main__":
#     load_dotenv(".devcontainer/devcontainer.env")
#     connection_str = os.getenv("CONNECTION_STR")
#     db = Database(connection_str)
#     db.connect()
#     db.close()