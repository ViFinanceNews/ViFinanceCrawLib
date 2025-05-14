import psycopg2
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime

class Database:
    def __init__(self, database_url):
        """
        Initializes the database connection.

        Parameters:
        - connection_string (str): CockroachDB connection string.
        """
        self.database_url = database_url
        self.conn = None

    def connect(self):
        """Establishes a database connection."""
        try:
            self.conn = psycopg2.connect(self.database_url)
            print("✅ Connected to CockroachDB successfully!")
        except psycopg2.Error as e:
            print(f"❌ Database connection failed: {e}")

    def create_table(self, table_name, schema):
        """
        Creates a table if it does not exist.

        Parameters:
        - table_name (str): The name of the table to create.
        - schema (dict): Dictionary where keys are column names and values are SQL data types.
        """
        if not self.conn:
            print("⚠️ No database connection.")
            return

        try:
            cursor = self.conn.cursor()
            columns = ", ".join([f"{col} {dtype}" for col, dtype in schema.items()])
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
            cursor.execute(query)
            self.conn.commit()
            cursor.close()
            print(f"✅ Table '{table_name}' is ready.")
        except psycopg2.Error as e:
            print(f"❌ Error creating table '{table_name}': {e}")

    def check_table_exists(self, table_name,schema_name='public'):
        """
        Checks if a table exists in CockroachDB.

        Parameters:
        - schema_name (str): Schema name (e.g., 'public')
        - table_name (str): Table name to check

        Returns:
        - True if the table exists, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            query = """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """
            cursor.execute(query, (schema_name, table_name))
            table_exists = cursor.fetchone() is not None
            cursor.close()
            return table_exists
        except Exception as e:
            print(f"❌ Error checking table existence: {e}")
            return False

    def insert_record(self, table_name, record):
        """Inserts a record into the specified table."""
        if not self.conn:
            print("⚠️ No database connection.")
            return

        if not record:
            print("⚠️ Empty record. Nothing to insert.")
            return

        try:
            cursor = self.conn.cursor()
            columns = ", ".join(record.keys())
            placeholders = ", ".join(["%s"] * len(record))
            values = tuple(record.values())
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, values)
            self.conn.commit()
            cursor.close()
            print(f"✅ Record inserted into '{table_name}'.")
        except psycopg2.Error as e:
            print(f"❌ Insert failed: {e}")

    def insert_records_bulk(self, table_name, records):
        """Performs a bulk insert into the specified table."""
        if not self.conn:
            print("⚠️ No database connection.")
            return

        if not records:
            print("⚠️ No records to insert.")
            return

        try:
            cursor = self.conn.cursor()
            columns = ", ".join(records[0].keys())
            placeholders = ", ".join(["%s"] * len(records[0]))
            values = [tuple(r.values()) for r in records]
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.executemany(query, values)
            self.conn.commit()
            cursor.close()
            print(f"✅ {len(records)} records inserted into '{table_name}'.")
        except psycopg2.Error as e:
            print(f"❌ Bulk insert failed: {e}")

    def execute_query(self, query, params=None, fetch_one=False, fetch_all=False, commit=False):
        """
            Executes a raw SQL query with optional fetch and commit options.

            Parameters:
            - query (str): The SQL query string to be executed.
            - params (tuple or list, optional): Parameters to be passed with the SQL query.
            - fetch_one (bool): If True, fetches a single row from the result.
            - fetch_all (bool): If True, fetches all rows from the result.
            - commit (bool): If True, commits the transaction after executing the query.

            Returns:
            - The result of the query if fetch_one or fetch_all is True, otherwise None.
        """
        if not self.conn:
            print("⚠️ No database connection.")
            return None

        if fetch_one and fetch_all:
            raise ValueError("⚠️ fetch_one and fetch_all cannot both be True.")

        try:
            print(query)
            with self.conn.cursor() as cursor:
                cursor.execute(query, params) if params else cursor.execute(query)

                result = None
                if fetch_one:
                    result = cursor.fetchone()
                elif fetch_all:
                    result = cursor.fetchall() or []

                if commit:
                    self.conn.commit()
                
                # Print the result for debugging
                

                return result
        except psycopg2.Error as e:
            print(f"❌ Query execution failed: {e}\nQuery: {query}")
            return None

    def show_data(self, table_name, columns=None, where_clause=None, limit=None):
        """Displays data from a specific table, with optional filters and limits."""
        if not columns:
            print("❌ Fail to display: no columns provided.")
            return

        try:
            # Join column names
            columns_str = ", ".join(columns)

            # Build query
            query = f"SELECT {columns_str} FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            if limit:
                query += f" LIMIT {limit}"

            # Debug print of the final SQL query (optional)
            print(f"🔍 Running query: {query}")

            # Execute query
            result = self.execute_query(query, fetch_all=True)

            # Display results
            if result:
                print(f"📋 Displaying data from table '{table_name}':")
                for row in result:
                    print(row)
            else:
                print("⚠️ No data found or query failed.")

        except Exception as e:
            print(f"❌ Error while displaying data from '{table_name}': {e}")

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("🔌 Connection closed.")
