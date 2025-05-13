#!/usr/bin/env python3
"""
PostgreSQL Database MCP Server (Simplified Schema Dump)

This MCP server provides tools for:
1. Connecting to a PostgreSQL database using a connection string.
2. On connection, fetching schema (tables, columns, types) and sample data for all tables.
3. Saving this schema and data dump to a Markdown file.
4. Initializing and retrieving this dump.
5. Executing SQL queries on the connected database.
6. Retrieving query results and table-specific information.
"""

import os
import json
import psycopg2
import psycopg2.extras # For DictCursor
import psycopg2.sql # For safe SQL identifiers
import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv # Added for .env loading

# MCP SDK
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

# Create an MCP server
mcp = FastMCP("PostgreSQL Database Server (Simple Dump)")

# Database connections storage
db_connections: Dict[str, 'PostgreSQLConnection'] = {}

class PostgreSQLConnection:
    """PostgreSQL database connection class."""
    
    def __init__(self, connection_string: str, db_name: str):
        """Initialize a database connection."""
        self.connection_string = connection_string
        self.db_name = db_name
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to the database."""
        try:
            self.connection = psycopg2.connect(self.connection_string)
            self.connection.autocommit = False 
            self.connected = True
            return True
        except Exception as e:
            print(f"Error connecting to PostgreSQL database '{self.db_name}': {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from the database."""
        try:
            if self.connection:
                self.connection.close()
            self.connected = False
            return True
        except Exception as e:
            print(f"Error disconnecting from PostgreSQL database '{self.db_name}': {e}")
            return False
    
    def get_tables(self, schema_name: str = 'public') -> List[str]:
        """Get a list of all tables in the specified schema."""
        if not self.connected or not self.connection:
            return []
        
        try:
            with self.connection.cursor() as cursor:
                query = psycopg2.sql.SQL("""
                    SELECT tablename 
                    FROM pg_catalog.pg_tables 
                    WHERE schemaname = %s ORDER BY tablename;
                """)
                cursor.execute(query, (schema_name,))
                tables = [row[0] for row in cursor.fetchall()]
                return tables
        except Exception as e:
            print(f"Error getting tables from PostgreSQL schema '{schema_name}' in db '{self.db_name}': {e}")
            return []

    def get_table_ddl(self, table_name: str, schema_name: str = 'public') -> str:
        if not self.connected or not self.connection:
            return f"-- Not connected to database {self.db_name} to get DDL for {schema_name}.{table_name}"
        
        ddl_retrieved = ""
        error_message = ""

        try:
            with self.connection.cursor() as cursor:
                # Attempt to get full DDL using pg_get_tabledef
                # This function might not exist or user may not have permissions
                query_pg_get_tabledef = psycopg2.sql.SQL("""
                    SELECT pg_catalog.pg_get_tabledef(c.oid)
                    FROM pg_catalog.pg_class c
                    LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relkind IN ('r', 'v', 'm', 'p', 'f') -- 'f' for foreign table
                    AND c.relname = %s
                    AND n.nspname = %s;
                """)
                cursor.execute(query_pg_get_tabledef, (table_name, schema_name))
                result = cursor.fetchone()
                if result and result[0]:
                    ddl_retrieved = result[0]
                else:
                    error_message += f"-- pg_get_tabledef did not return DDL for {schema_name}.{table_name}. "
        except psycopg2.Error as e:
            # Catch psycopg2 specific errors, often permission or function not found
            error_message += f"-- Error using pg_get_tabledef for {schema_name}.{table_name}: {str(e).strip()}. "
            if self.connection and not self.connection.closed:
                 self.connection.rollback() # Rollback any transaction state

        # Fallback: Construct DDL from information_schema if pg_get_tabledef failed
        if not ddl_retrieved:
            error_message += "Attempting fallback DDL construction. "
            cols_info_result = self.get_table_column_info(table_name, schema_name)
            
            if isinstance(cols_info_result, str): # Error string returned
                 return f"-- Fallback DDL construction failed for {schema_name}.{table_name} because column info could not be retrieved: {cols_info_result}"

            if not cols_info_result: # Empty list returned
                return f"-- Fallback DDL construction failed for {schema_name}.{table_name}: No column information found."

            try:
                # Ensure connection is available for Identifier.as_string
                if not self.connection or self.connection.closed:
                    return f"-- Fallback DDL construction failed for {schema_name}.{table_name}: Connection closed before formatting identifiers."

                col_defs = []
                for c in cols_info_result:
                    col_name_sql = psycopg2.sql.Identifier(c['name']).as_string(self.connection)
                    col_type_sql = c['type'] # udt_name is usually sufficient
                    col_def_str = f"    {col_name_sql} {col_type_sql}"
                    if not c.get('is_nullable', True): # is_nullable might be missing
                        col_def_str += " NOT NULL"
                    if c.get('default_value') is not None:
                        col_def_str += f" DEFAULT {c['default_value']}"
                    col_defs.append(col_def_str)
                
                table_name_sql = psycopg2.sql.Identifier(table_name).as_string(self.connection)
                schema_name_sql = psycopg2.sql.Identifier(schema_name).as_string(self.connection)
                ddl_retrieved = f"CREATE TABLE {schema_name_sql}.{table_name_sql} (\n" + ",\n".join(col_defs) + "\n);"
                error_message = "" # Clear previous errors if fallback succeeded
            except Exception as e_fallback:
                error_message += f"Error during fallback DDL construction for {schema_name}.{table_name}: {str(e_fallback).strip()}."
                ddl_retrieved = "" # Ensure no partial DDL is returned

        if error_message and not ddl_retrieved:
            return f"-- Could not retrieve DDL for {schema_name}.{table_name}. Errors: {error_message.strip()}"
        elif error_message and ddl_retrieved: # DDL retrieved but there were non-fatal errors (e.g. pg_get_tabledef failed but fallback worked)
            return f"-- Note: There were issues during DDL retrieval for {schema_name}.{table_name}: {error_message.strip()}\n{ddl_retrieved}"
        
        return ddl_retrieved if ddl_retrieved else f"-- Unknown error: Failed to retrieve DDL for {schema_name}.{table_name} and no specific errors captured."

    def get_table_column_info(self, table_name: str, schema_name: str = 'public') -> Union[List[Dict[str, Any]], str]:
        if not self.connected or not self.connection:
            return f"-- Not connected to database {self.db_name} to get column info for {schema_name}.{table_name}"
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = psycopg2.sql.SQL("""
                    SELECT column_name, udt_name, data_type, character_maximum_length, column_default, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = %s ORDER BY ordinal_position;
                """)
                cursor.execute(query, (table_name, schema_name))
                columns_data = cursor.fetchall()
                if not columns_data:
                    return f"-- No columns found for table {schema_name}.{table_name} in information_schema.columns. The table might be empty, a view with no column definitions accessible, or the user lacks permissions."
                
                return [{
                    "name": r["column_name"], "type": r["udt_name"], "data_type_standard": r["data_type"],
                    "max_length": r["character_maximum_length"], "default_value": r["column_default"],
                    "is_nullable": r["is_nullable"] == "YES"
                } for r in columns_data]
        except psycopg2.Error as e:
            error_msg = f"-- Error querying information_schema.columns for {schema_name}.{table_name}: {str(e).strip()}"
            print(f"Error getting column info for {schema_name}.{table_name} in db '{self.db_name}': {e}")
            if self.connection and not self.connection.closed:
                self.connection.rollback()
            return error_msg
        except Exception as e: # Catch any other unexpected errors
            error_msg = f"-- Unexpected error getting column info for {schema_name}.{table_name}: {str(e).strip()}"
            print(error_msg)
            return error_msg
    
    def execute_query(self, query: str, params: Optional[List[Any]] = None, row_limit: Optional[int] = None) -> Dict[str, Any]:
        if not self.connected or not self.connection:
            return {"error": f"Not connected to PostgreSQL database '{self.db_name}'"}
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                start_time = datetime.datetime.now()
                cursor.execute(query, params if params else None)
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                
                if query.strip().lower().startswith("select"):
                    columns = [desc[0] for desc in cursor.description]
                    rows_tuples = cursor.fetchmany(row_limit) if row_limit is not None else cursor.fetchall()
                    rows = [dict(row) for row in rows_tuples]
                    return {"columns": columns, "rows": rows, "row_count": len(rows), "execution_time": execution_time}
                else:
                    self.connection.commit()
                    return {"affected_rows": cursor.rowcount, "execution_time": execution_time, "message": "Query executed and committed."}
        except Exception as e:
            if self.connection: self.connection.rollback()
            return {"error": str(e)}
    
    def get_table_data_df(self, table_name: str, schema_name: str = 'public', limit: Optional[int] = 5) -> Union[pd.DataFrame, str]:
        """Get sample data from a table as a pandas DataFrame, or an error string."""
        if not self.connected or not self.connection:
            return f"-- Not connected to database {self.db_name} to get sample data for {schema_name}.{table_name}"
        try:
            # Ensure connection is valid for as_string
            if not self.connection or self.connection.closed:
                 return f"-- Connection closed before querying sample data for {schema_name}.{table_name}"

            schema_identifier = psycopg2.sql.Identifier(schema_name).as_string(self.connection)
            table_identifier = psycopg2.sql.Identifier(table_name).as_string(self.connection)
            
            sql_query_str = f"SELECT * FROM {schema_identifier}.{table_identifier}"
            if limit is not None:
                sql_query_str += f" LIMIT {limit}"
            
            df = pd.read_sql_query(sql_query_str, self.connection)
            if df.empty:
                # Check if table actually has rows or if it's just empty
                with self.connection.cursor() as cursor:
                    cursor.execute(f"SELECT EXISTS (SELECT 1 FROM {schema_identifier}.{table_identifier} LIMIT 1)")
                    table_has_rows = cursor.fetchone()[0]
                    if not table_has_rows:
                        return pd.DataFrame() # Return empty DataFrame if table is genuinely empty
                    else:
                        # This case is unlikely if read_sql_query worked but returned empty for a non-empty table
                        return f"-- Fetched an empty DataFrame for {schema_name}.{table_name} but the table appears to have rows. Limit might be too small or other query issue."
            return df
        except psycopg2.Error as e:
            error_msg = f"-- Error executing query for sample data from {schema_name}.{table_name}: {str(e).strip()}"
            print(f"Error getting DataFrame for {schema_name}.{table_name} in db '{self.db_name}': {e}")
            if self.connection and not self.connection.closed:
                self.connection.rollback()
            return error_msg
        except Exception as e: # Catch other pandas or unexpected errors
            error_msg = f"-- Unexpected error getting DataFrame for {schema_name}.{table_name}: {str(e).strip()}"
            print(error_msg)
            return error_msg

    def generate_schema_and_data_markdown(self, schema_name: str = 'public', sample_rows: int = 5) -> str:
        """
        Generates a Markdown document containing schema and sample data for all tables.
        Handles cases where DDL, column info, or sample data retrieval might fail.
        """
        if not self.connected or not self.connection:
            return f"# Schema and Data Dump Error for '{self.db_name}'\n-- Not connected to database."

        tables = self.get_tables(schema_name)
        if not tables: # Could be an error or genuinely no tables
            # Check connection again, as get_tables might return [] if not connected.
            if not self.connected or not self.connection:
                 return f"# Schema and Data Dump for '{self.db_name}' (Schema: '{schema_name}')\n-- Error: Not connected when trying to list tables."
            return f"# Schema and Data Dump for '{self.db_name}' (Schema: '{schema_name}')\n-- No tables found in schema '{schema_name}' or error listing tables."

        markdown_output = [f"# Schema and Data Dump for Database: '{self.db_name}', Schema: '{schema_name}'"]
        markdown_output.append(f"Generated on: {datetime.datetime.now().isoformat()}\n")

        for table_name in tables:
            markdown_output.append(f"## Table: `{table_name}`\n")
            
            # Get DDL (CREATE TABLE statement)
            ddl_result = self.get_table_ddl(table_name, schema_name)
            markdown_output.append("### DDL (CREATE TABLE Statement):\n")
            markdown_output.append(f"```sql\n{ddl_result}\n```\n") # ddl_result is already a string, potentially with error messages

            # Get Column Information
            columns_info_result = self.get_table_column_info(table_name, schema_name)
            markdown_output.append("### Columns:\n")
            if isinstance(columns_info_result, str): # Error message string
                markdown_output.append(columns_info_result)
            elif columns_info_result: # List of dicts
                try:
                    cols_df = pd.DataFrame(columns_info_result)
                    # Select common columns, ensure they exist to prevent KeyErrors
                    display_cols = [col for col in ["name", "type", "is_nullable", "default_value", "max_length", "data_type_standard"] if col in cols_df.columns]
                    if display_cols:
                        markdown_output.append(cols_df[display_cols].to_markdown(index=False))
                    else:
                        markdown_output.append("-- Column information retrieved but no standard columns to display or DataFrame is malformed.")
                except Exception as e_cols_df:
                    markdown_output.append(f"-- Error formatting column information into Markdown: {str(e_cols_df)}")
            else: # Empty list, but not an error string
                markdown_output.append("-- No column information retrieved (empty list).")
            markdown_output.append("\n")

            # Get Sample Data
            markdown_output.append(f"### Sample Data (Top {sample_rows} rows):\n")
            sample_data_result = self.get_table_data_df(table_name, schema_name, limit=sample_rows)
            if isinstance(sample_data_result, str): # Error message string
                markdown_output.append(sample_data_result)
            elif not sample_data_result.empty: # DataFrame with data
                try:
                    markdown_output.append(sample_data_result.to_markdown(index=False))
                except Exception as e_sample_df:
                    markdown_output.append(f"-- Error formatting sample data into Markdown: {str(e_sample_df)}")
            else: # Empty DataFrame, no error string
                markdown_output.append(f"-- No sample data found in table `{table_name}` or table is empty (returned empty DataFrame).")
            markdown_output.append("\n---\n")
        
        return "\n".join(markdown_output)

#############################################################
# MCP Tools                                                 #
#############################################################

@mcp.tool()
def connect_to_postgres(connection_string: str, db_name: str, schema_folder_path: str) -> Dict[str, Any]:
    """
    Connect to a PostgreSQL database, fetch schema and sample data for all tables,
    and save it to a Markdown file in the specified folder.
    Example connection string: "postgresql://user:password@host:port/dbname"
    
    Args:
        connection_string: PostgreSQL connection string.
        db_name: A unique name to identify this database connection.
        schema_folder_path: The full path to the folder where the schema dump file should be saved.
        
    Returns:
        Dictionary with connection result, including path to the schema and data dump file.
    """
    if db_name in db_connections:
        return {"success": False, "message": f"Connection '{db_name}' already exists."}
    
    connection = PostgreSQLConnection(connection_string, db_name)
    if not connection.connect():
        return {"success": False, "message": f"Failed to connect to PostgreSQL for '{db_name}'."}
    
    db_connections[db_name] = connection
    
    # Define path for schema dump using the provided folder path
    dump_dir = Path(schema_folder_path)
    dump_file = dump_dir / f"{db_name}_schema_data.md"
    schema_dump_md_path = str(dump_file.resolve())

    if dump_file.exists():
        print(f"Schema dump file already exists for '{db_name}' at {schema_dump_md_path}. Skipping generation.")
        schema_dump_status = f"Existing schema dump file found at {schema_dump_md_path}"
    else:
        try:
            print(f"Generating schema and data dump for '{db_name}'...")
            markdown_dump = connection.generate_schema_and_data_markdown(schema_name='public', sample_rows=5)
            
            dump_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            
            with open(dump_file, "w", encoding="utf-8") as f:
                f.write(markdown_dump)
            schema_dump_status = f"Successfully generated and saved to {schema_dump_md_path}"
            print(schema_dump_status)
        except Exception as e:
            schema_dump_status = f"Failed to generate schema and data dump: {e}"
            print(schema_dump_status)
            # If generation fails, path might not be valid, so clear it
            schema_dump_md_path = "" 

    return {
        "success": True,
        "message": f"Connected to PostgreSQL '{db_name}'. Schema and data dump process completed.",
        "db_name": db_name,
        "schema_dump_status": schema_dump_status,
        "schema_data_file_path": schema_dump_md_path if schema_dump_md_path else None
    }

@mcp.tool()
def initialize_db_summary(db_name: str) -> Dict[str, Any]:
    """
    Reads and returns the content of the pre-generated schema and data dump Markdown file.
    
    Args:
        db_name: The name of the database connection for which to load the dump.
        
    Returns:
        Dictionary with the dump content or an error message.
    """
    # Path matches the one used in connect_to_postgres
    dump_file_path = Path("mcp_servers") / "db_schemas" / f"{db_name}_schema_data.md"
    
    if not dump_file_path.exists():
        return {
            "success": False,
            "message": f"Schema and data dump file not found: {dump_file_path}. Ensure 'connect_to_postgres' was called successfully first."
        }
    
    try:
        with open(dump_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "success": True,
            "db_name": db_name,
            "schema_data_file_path": str(dump_file_path.resolve()),
            "schema_data_content": content
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error reading schema and data dump file '{dump_file_path}': {e}"
        }

@mcp.tool()
def disconnect_postgres(db_name: str) -> Dict[str, Any]:
    if db_name not in db_connections:
        return {"success": False, "message": f"No connection '{db_name}'."}
    if db_connections[db_name].disconnect():
        del db_connections[db_name]
        return {"success": True, "message": f"Disconnected from '{db_name}'."}
    return {"success": False, "message": f"Failed to disconnect '{db_name}'."}

@mcp.tool()
def list_postgres_connections() -> Dict[str, Any]:
    return {"connections": [{"name": name, "status": "connected" if conn.connected else "disconnected"} 
                           for name, conn in db_connections.items()]}

@mcp.tool()
def execute_postgres_query(db_name: str, query: str, params: Optional[List[Any]] = None, row_limit: int = 100) -> Dict[str, Any]:
    if db_name not in db_connections:
        return {"success": False, "message": f"No connection '{db_name}'."}
    conn = db_connections[db_name]
    actual_row_limit = row_limit if query.strip().lower().startswith("select") else None
    result = conn.execute_query(query, params=params, row_limit=actual_row_limit)
    
    if "error" in result: return {"success": False, "message": result["error"]}
    
    formatted_res = {"success": True, "result": result}
    if query.strip().lower().startswith("select") and "columns" in result and "rows" in result:
        try:
            df = pd.DataFrame(result["rows"], columns=result["columns"])
            formatted_res["formatted_table"] = df.to_markdown(index=False)
        except Exception as e: formatted_res["formatted_table_error"] = str(e)
    return formatted_res

@mcp.tool()
def get_postgres_table_info(db_name: str, table_name: str, schema_name: str = 'public') -> Dict[str, Any]:
    if db_name not in db_connections: return {"success": False, "message": f"No connection '{db_name}'."}
    conn = db_connections[db_name]
    columns = conn.get_table_column_info(table_name, schema_name)
    if not columns: return {"success": False, "message": f"Error getting columns for '{schema_name}.{table_name}'."}
    
    sample_df = conn.get_table_data_df(table_name, schema_name, limit=5)
    sample_data_md = sample_df.to_markdown(index=False) if not sample_df.empty else "No sample data or table is empty."
    
    ddl = conn.get_table_ddl(table_name, schema_name)

    return {"success": True, "db_name": db_name, "schema_name": schema_name, "table_name": table_name, 
            "columns": columns, 
            "ddl": ddl if ddl else f"-- Could not retrieve DDL for {schema_name}.{table_name}",
            "sample_data_markdown": sample_data_md}

@mcp.tool()
def get_postgres_table_data(db_name: str, table_name: str, schema_name: str = 'public', limit: int = 100) -> Dict[str, Any]:
    if db_name not in db_connections: return {"success": False, "message": f"No connection '{db_name}'."}
    conn = db_connections[db_name]
    
    df = conn.get_table_data_df(table_name, schema_name, limit=limit)
    if df.empty and not conn.get_table_column_info(table_name, schema_name):
         return {"success": False, "message": f"Table {schema_name}.{table_name} not found or is empty and schema could not be read."}

    return {"success": True, "db_name": db_name, "schema_name": schema_name, "table_name": table_name,
            "row_count": len(df), "data_markdown": df.to_markdown(index=False) if not df.empty else "Table is empty or data fetch failed."}

# Resources
@mcp.resource("postgres-schema://{db_name}/{schema_name}")
def get_postgres_schema_resource(db_name: str, schema_name: str) -> Dict[str, Any]:
    if db_name not in db_connections or not db_connections[db_name].connected:
        return {"error": f"Connection '{db_name}' not active."}
    conn = db_connections[db_name]
    tables = conn.get_tables(schema_name)
    schema_details = {}
    for t in tables:
        schema_details[t] = {
            "ddl": conn.get_table_ddl(t, schema_name),
            "columns": conn.get_table_column_info(t, schema_name)
        }
    return {"db_name": db_name, "schema_name": schema_name, "tables": schema_details}

@mcp.resource("postgres-table://{db_name}/{schema_name}/{table_name}")
def get_postgres_table_resource(db_name: str, schema_name: str, table_name: str) -> Dict[str, Any]:
    if db_name not in db_connections or not db_connections[db_name].connected:
        return {"error": f"Connection '{db_name}' not active."}
    conn = db_connections[db_name]
    columns = conn.get_table_column_info(table_name, schema_name)
    sample_df = conn.get_table_data_df(table_name, schema_name, limit=5)
    ddl = conn.get_table_ddl(table_name, schema_name)
    return {"db_name": db_name, "schema_name": schema_name, "table_name": table_name, 
            "ddl": ddl, "columns": columns,
            "sample_data": sample_df.to_dict(orient='records') if not sample_df.empty else []}

@mcp.resource("postgres-connections://list")
def get_postgres_connections_resource() -> Dict[str, Any]:
    return {"connections": [{"name": n, "status": "connected" if c.connected else "disconnected"} for n, c in db_connections.items()]}

if __name__ == "__main__":
    print("Starting PostgreSQL MCP Server (Simple Dump version)...")
    mcp.run()
