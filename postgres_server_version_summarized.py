#!/usr/bin/env python3
"""
PostgreSQL Database MCP Server

This MCP server provides tools for:
1. Connecting to a PostgreSQL database using a connection string
2. Executing SQL queries on the connected database
3. Retrieving query results and schema information (including CREATE TABLE DDLs)
4. Generating an LLM-based summary of the database schema upon connection.
5. Initializing and retrieving this summary.
"""

import os
import json
import psycopg2
import psycopg2.extras # For DictCursor
import psycopg2.sql # For safe SQL identifiers
import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv # Added for .env loading

# MCP SDK
from mcp.server.fastmcp import FastMCP

# For LLM interaction
# from openai import OpenAI # Replaced for Gemini
import google.generativeai as genai # Added for Gemini

# Load environment variables from .env file
load_dotenv()

# Create an MCP server
mcp = FastMCP("PostgreSQL Database Server")

# Database connections storage
db_connections: Dict[str, 'PostgreSQLConnection'] = {}

# --- Code from message.txt (adapted) ---

def generate_column_descriptions(df: pd.DataFrame, pruned_itemsets: Optional[pd.DataFrame] = None, constant_columns: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Generate descriptions for columns based on data values and patterns.
    Adapted to work without pruned_itemsets for general table description.
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping column names to descriptions
    """
    column_descriptions = {}
    constant_cols = constant_columns or {}
    
    all_itemset_columns = set()
    if pruned_itemsets is not None and not pruned_itemsets.empty:
        for _, row in pruned_itemsets.iterrows():
            itemset = row['itemsets']
            for item in itemset: # Assuming itemset is a list of strings like "col_name___value"
                if isinstance(item, str) and "___" in item:
                    col = item.split("___")[0]
                    all_itemset_columns.add(col)
    else:
        # If no pruned_itemsets, consider all columns in the DataFrame
        all_itemset_columns = set(df.columns)

    for col in df.columns:
        if col not in all_itemset_columns: # This condition might be too restrictive if we want all df columns
            continue
            
        if col.lower() in ['id', 'key', 'index', 'row_id', 'record_id', 'pk', 'primary_key']: # Expanded skip list
            column_descriptions[col] = "Likely primary key or identifier column."
            continue
            
        col_type = str(df[col].dtype)
        desc = f"Column '{col}' of type '{col_type}'." # Default description
        
        try: # Add try-except for robustness with diverse data
            unique_values = df[col].dropna().unique()
            n_unique = len(unique_values)
            
            example_values = [str(v) for v in unique_values[:5]]
            examples_str = ", ".join(example_values)
            if len(unique_values) > 5:
                examples_str += ", etc."
                
            is_categorical = False
            # Refined categorical detection
            if n_unique == 0 and df[col].isnull().all():
                desc = f"Column '{col}' (type: {col_type}) contains only NULL values."
            elif col_type in ['object', 'string', 'category', 'bool'] or 'bool' in col_type.lower():
                is_categorical = True
            elif 'int' in col_type or 'float' in col_type:
                # Consider a numeric column categorical if it has few unique values relative to total non-null rows
                if n_unique > 0 and (n_unique <= 10 or (df[col].count() > 0 and n_unique / df[col].count() < 0.1 and n_unique <= 20)):
                    is_categorical = True
            
            if col in constant_cols:
                desc = f"Column '{col}' (type: {col_type}) has a constant value: '{constant_cols[col]}'."
            elif is_categorical:
                desc = f"Categorical column '{col}' (type: {col_type}) with {n_unique} unique values. Examples: {examples_str}."
            elif 'int' in col_type:
                min_val = df[col].min()
                max_val = df[col].max()
                desc = f"Integer column '{col}' (type: {col_type}) with values from {min_val} to {max_val}. Examples: {examples_str}."
            elif 'float' in col_type:
                min_val = df[col].min()
                max_val = df[col].max()
                desc = f"Numeric column '{col}' (type: {col_type}) with values from {min_val:.2f} to {max_val:.2f}. Examples: {examples_str}."
            elif 'date' in col_type.lower() or 'time' in col_type.lower():
                try:
                    min_dt = pd.to_datetime(df[col]).min()
                    max_dt = pd.to_datetime(df[col]).max()
                    desc = f"Date/time column '{col}' (type: {col_type}) with values from {min_dt} to {max_dt}. Examples: {examples_str}."
                except Exception:
                    desc = f"Date/time-like column '{col}' (type: {col_type}). Examples: {examples_str}."
            else: # Fallback for other types
                desc = f"Column '{col}' (type: {col_type}) with {n_unique} unique values. Examples: {examples_str}."
        except Exception as e:
            desc = f"Column '{col}' (type: {col_type}). Error during analysis: {e}. Examples: {', '.join([str(v) for v in df[col].dropna().unique()[:3]])}..."


        column_descriptions[col] = desc
            
    return column_descriptions

def generate_database_schema_summary_llm(
    all_table_column_descriptions: Dict[str, Dict[str, str]],
    db_name: str,
    # api_key parameter removed, will be fetched from environment
    model_name: str = "gemini-2.5-pro-preview-05-06"
) -> str:
    """
    Uses Google Gemini to generate a narrative summary of the database schema.
    API key is fetched from environment variable GEMINI_API_KEY.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error_msg = "GEMINI_API_KEY not found in environment variables."
        print(f"Error for '{db_name}': {error_msg}")
        return f"Error: {error_msg}"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error configuring Gemini client for '{db_name}': {e}")
        return f"Error configuring Gemini client: {e}"

    formatted_descriptions = f"DATABASE SCHEMA ANALYSIS for '{db_name}':\n\n"
    for table_name, descriptions in all_table_column_descriptions.items():
        formatted_descriptions += f"TABLE: {table_name}\n"
        if descriptions:
            for col, desc in descriptions.items():
                formatted_descriptions += f"  - {col}: {desc}\n"
        else:
            formatted_descriptions += "  - (No detailed column descriptions generated for this table, might be empty or only identifiers).\n"
        formatted_descriptions += "\n"

    prompt = f"""
    You are a data architect and database analyst. Based on the following table and column descriptions for a PostgreSQL database named '{db_name}', provide a concise and insightful summary of the database schema.

    Focus on:
    1. The likely purpose or domain of the database.
    2. The main entities represented by the tables.
    3. Potential relationships between tables (e.g., one-to-many, many-to-many, lookups) based on column names and descriptions.
    4. Any particularly interesting or complex tables or columns you observe.
    5. Overall structure and any apparent design patterns or conventions.

    TABLE AND COLUMN DESCRIPTIONS:
    {formatted_descriptions}

    Provide your analysis in Markdown format.
    Start with a high-level overview, then discuss key tables and potential relationships.
    Be analytical and inferential, but clearly state when you are making assumptions.
    """
    
    generation_config = genai.types.GenerationConfig(
        temperature=0.3,
        max_output_tokens=5000 # Increased slightly for potentially more verbose Gemini
    )

    try:
        # For Gemini, the prompt is passed directly.
        # System prompt/role is often handled via initial instructions or few-shot examples if needed,
        # or by structuring the user prompt carefully. Here, the user prompt is quite descriptive.
        response = model.generate_content(prompt, generation_config=generation_config)
        summary = response.text
        return summary if summary else "Gemini LLM failed to generate a summary."
    except Exception as e:
        print(f"Error during Gemini LLM schema summary generation for '{db_name}': {e}")
        return f"Error generating schema summary with Gemini: {e}"


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
            self.connected = False # Ensure connected is false on error
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

    def get_table_ddl(self, table_name: str, schema_name: str = 'public') -> Optional[str]:
        if not self.connected or not self.connection:
            return f"-- Not connected to database {self.db_name}"
        try:
            with self.connection.cursor() as cursor:
                query_pg_get_tabledef = psycopg2.sql.SQL("""
                    SELECT pg_catalog.pg_get_tabledef(c.oid, true)
                    FROM pg_catalog.pg_class c
                    LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relkind IN ('r', 'v', 'm', 'p') -- r: table, v: view, m: materialized view, p: partitioned table
                    AND c.relname = %s
                    AND n.nspname = %s;
                """)
                cursor.execute(query_pg_get_tabledef, (table_name, schema_name))
                result = cursor.fetchone()
                
                if result and result[0]:
                    return result[0]
                else:
                    # Fallback for basic DDL (simplified)
                    cols_info = self.get_table_column_info(table_name, schema_name)
                    if not cols_info: return f"-- Could not retrieve DDL for {schema_name}.{table_name}"
                    col_defs = [f"    {psycopg2.sql.Identifier(c['name']).as_string(self.connection)} {c['type']}" for c in cols_info]
                    return f"CREATE TABLE {psycopg2.sql.Identifier(schema_name).as_string(self.connection)}.{psycopg2.sql.Identifier(table_name).as_string(self.connection)} (\n" + ",\n".join(col_defs) + "\n);"
        except Exception as e:
            return f"-- Error retrieving DDL for {schema_name}.{table_name}: {e}"


    def get_table_column_info(self, table_name: str, schema_name: str = 'public') -> List[Dict[str, Any]]:
        if not self.connected or not self.connection: return []
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = psycopg2.sql.SQL("""
                    SELECT column_name, udt_name, data_type, character_maximum_length, column_default, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = %s ORDER BY ordinal_position;
                """)
                cursor.execute(query, (table_name, schema_name))
                return [{
                    "name": r["column_name"], "type": r["udt_name"], "data_type_standard": r["data_type"],
                    "max_length": r["character_maximum_length"], "default_value": r["column_default"],
                    "is_nullable": r["is_nullable"] == "YES"
                } for r in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting column info for {schema_name}.{table_name} in db '{self.db_name}': {e}")
            return []
    
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
    
    def get_table_data_df(self, table_name: str, schema_name: str = 'public', limit: Optional[int] = 5) -> pd.DataFrame:
        """Get sample data from a table as a pandas DataFrame."""
        if not self.connected or not self.connection: return pd.DataFrame()
        try:
            sql_query = psycopg2.sql.SQL("SELECT * FROM {}.{}").format(
                psycopg2.sql.Identifier(schema_name), psycopg2.sql.Identifier(table_name)
            )
            # Ensure params is a tuple or list for pd.read_sql_query
            params_for_query: Optional[tuple] = None
            if limit is not None:
                sql_query = psycopg2.sql.SQL(" ").join([sql_query, psycopg2.sql.SQL("LIMIT %s")])
                params_for_query = (limit,)
            
            return pd.read_sql_query(sql_query.as_string(self.connection), self.connection, params=params_for_query)
        except Exception as e:
            print(f"Error getting DataFrame for {schema_name}.{table_name} in db '{self.db_name}': {e}")
            return pd.DataFrame()

    def generate_database_summary_markdown(self, schema_name: str = 'public') -> str: # gemini_api_key parameter removed
        """Generates a Markdown summary of the database schema using LLM. API key is fetched from environment."""
        if not self.connected:
            return "# Database Summary Error\nNot connected to database."
        
        # API key will be fetched by generate_database_schema_summary_llm directly
        # No need to check for gemini_api_key here anymore

        tables = self.get_tables(schema_name)
        if not tables:
            return f"# Database Summary for {self.db_name}\nNo tables found in schema '{schema_name}'."

        all_table_col_descs: Dict[str, Dict[str, str]] = {}
        
        summary_intro = f"# AI-Generated Database Schema Summary for '{self.db_name}' (Schema: '{schema_name}')\n\n"
        summary_intro += f"Generated on: {datetime.datetime.now().isoformat()}\n\n"
        summary_intro += "This summary provides an overview of the database structure, inferred purposes of tables, and potential relationships based on table schemas and sample data analysis.\n\n"

        for table_name in tables:
            df_sample = self.get_table_data_df(table_name, schema_name, limit=10) # Fetch 10 rows for better description
            if not df_sample.empty:
                # Pass an empty DataFrame for pruned_itemsets as it's not applicable here
                col_descs = generate_column_descriptions(df_sample, pruned_itemsets=pd.DataFrame()) 
                all_table_col_descs[table_name] = col_descs
            else:
                # If table is empty, try to get column info from schema
                cols_info = self.get_table_column_info(table_name, schema_name)
                if cols_info:
                    all_table_col_descs[table_name] = {
                        col['name']: f"Column '{col['name']}' of type '{col['type']}' (Table might be empty or data fetch failed)."
                        for col in cols_info
                    }
                else:
                     all_table_col_descs[table_name] = {"info": "Could not retrieve column information for this empty table."}


        llm_summary = generate_database_schema_summary_llm(all_table_col_descs, self.db_name) # api_key argument removed
        
        # Combine intro with LLM summary
        final_summary = summary_intro + "## LLM Analysis (Gemini):\n" + llm_summary + "\n\n## Detailed Table Descriptions (from data analysis):\n"
        
        for table_name, descriptions in all_table_col_descs.items():
            final_summary += f"### Table: {table_name}\n"
            if descriptions:
                for col, desc in descriptions.items():
                    final_summary += f"- **{col}**: {desc}\n"
            else:
                final_summary += "- (No detailed column descriptions generated).\n"
            final_summary += "\n"
            
        return final_summary

#############################################################
# MCP Tools                                                 #
#############################################################

@mcp.tool()
def connect_to_postgres(connection_string: str, db_name: str, summary_folder_path: str) -> Dict[str, Any]: # gemini_api_key parameter removed
    """
    Connect to a PostgreSQL database, generate and save an AI summary of its schema using Gemini
    into the specified folder.
    The Gemini API key is expected to be in the GEMINI_API_KEY environment variable.
    Example connection string: "postgresql://user:password@host:port/dbname"
    
    Args:
        connection_string: PostgreSQL connection string.
        db_name: A unique name to identify this database connection.
        summary_folder_path: The full path to the folder where the schema summary file should be saved.
        
    Returns:
        Dictionary with connection result, including path to the schema summary file.
    """
    if db_name in db_connections:
        return {"success": False, "message": f"Connection '{db_name}' already exists."}
    
    connection = PostgreSQLConnection(connection_string, db_name)
    if not connection.connect():
        return {"success": False, "message": f"Failed to connect to PostgreSQL for '{db_name}'."}
    
    db_connections[db_name] = connection
    tables_public_schema = connection.get_tables(schema_name='public')
    
    summary_md_path = ""
    summary_generation_status = "Skipped (GEMINI_API_KEY not found in environment)."

    # Define summary file path using the provided folder path
    summary_dir = Path(summary_folder_path)
    summary_file = summary_dir / f"{db_name}_public_schema_summary.md"

    # Attempt to generate and save summary if API key is available in environment
    if os.getenv("GEMINI_API_KEY"):
        summary_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        if summary_file.exists():
            summary_md_path = str(summary_file.resolve())
            summary_generation_status = f"Summary already exists at {summary_md_path}. Skipping generation."
            print(summary_generation_status)
        else:
            try:
                print(f"Generating schema summary for '{db_name}' using Gemini (key from env)...")
                # gemini_api_key argument removed from generate_database_summary_markdown call
                markdown_summary = connection.generate_database_summary_markdown(schema_name='public') 
                
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write(markdown_summary)
                summary_md_path = str(summary_file.resolve())
                summary_generation_status = f"Successfully generated and saved to {summary_md_path}"
                print(summary_generation_status)
            except Exception as e:
                summary_generation_status = f"Failed to generate summary with Gemini: {e}"
                print(summary_generation_status)
    else:
        # Check if file exists even if API key is not set, to report its presence
        if summary_file.exists():
            summary_md_path = str(summary_file.resolve())
            summary_generation_status = f"Summary previously generated at {summary_md_path}. GEMINI_API_KEY not currently set."
        else:
            summary_generation_status = "Skipped (GEMINI_API_KEY not found and no pre-existing summary file)."
        print("Gemini API key not provided. Skipping schema summary generation if not already present.")

    return {
        "success": True,
        "message": f"Connected to PostgreSQL '{db_name}'. Found {len(tables_public_schema)} tables in 'public' schema.",
        "db_name": db_name,
        "public_schema_tables": tables_public_schema,
        "schema_summary_status": summary_generation_status,
        "schema_summary_file": summary_md_path if summary_md_path else None
    }

@mcp.tool()
def initialize_db_summary(db_name: str) -> Dict[str, Any]:
    """
    Reads and returns the content of the pre-generated database schema summary file.
    
    Args:
        db_name: The name of the database connection for which to load the summary.
        
    Returns:
        Dictionary with the summary content or an error message.
    """
    summary_file_path = Path("sql-generator") / "summary" / f"{db_name}_public_schema_summary.md"
    
    if not summary_file_path.exists():
        return {
            "success": False,
            "message": f"Summary file not found: {summary_file_path}. Ensure 'connect_to_postgres' was called first with an API key."
        }
    
    try:
        with open(summary_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "success": True,
            "db_name": db_name,
            "summary_file_path": str(summary_file_path.resolve()),
            "summary_content": content
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error reading summary file '{summary_file_path}': {e}"
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
    
    return {"success": True, "db_name": db_name, "schema_name": schema_name, "table_name": table_name, 
            "columns": columns, "sample_data_markdown": sample_data_md}

@mcp.tool()
def get_postgres_table_data(db_name: str, table_name: str, schema_name: str = 'public', limit: int = 100) -> Dict[str, Any]:
    if db_name not in db_connections: return {"success": False, "message": f"No connection '{db_name}'."}
    conn = db_connections[db_name]
    
    df = conn.get_table_data_df(table_name, schema_name, limit=limit)
    if df.empty and not conn.get_table_column_info(table_name, schema_name): # Check if table actually exists if df is empty
         return {"success": False, "message": f"Table {schema_name}.{table_name} not found or is empty and schema could not be read."}

    return {"success": True, "db_name": db_name, "schema_name": schema_name, "table_name": table_name,
            "row_count": len(df), "data_markdown": df.to_markdown(index=False) if not df.empty else "Table is empty or data fetch failed."}

# Resources (simplified for brevity, ensure they use updated connection methods if needed)
@mcp.resource("postgres-schema://{db_name}/{schema_name}")
def get_postgres_schema_resource(db_name: str, schema_name: str) -> Dict[str, Any]:
    if db_name not in db_connections or not db_connections[db_name].connected:
        return {"error": f"Connection '{db_name}' not active."}
    conn = db_connections[db_name]
    tables = conn.get_tables(schema_name)
    return {"db_name": db_name, "schema_name": schema_name, 
            "table_ddls": {t: conn.get_table_ddl(t, schema_name) for t in tables}}

@mcp.resource("postgres-table://{db_name}/{schema_name}/{table_name}")
def get_postgres_table_resource(db_name: str, schema_name: str, table_name: str) -> Dict[str, Any]:
    if db_name not in db_connections or not db_connections[db_name].connected:
        return {"error": f"Connection '{db_name}' not active."}
    conn = db_connections[db_name]
    columns = conn.get_table_column_info(table_name, schema_name)
    sample_df = conn.get_table_data_df(table_name, schema_name, limit=5)
    return {"db_name": db_name, "schema_name": schema_name, "table_name": table_name, "columns": columns,
            "sample_data": sample_df.to_dict(orient='records') if not sample_df.empty else []}

@mcp.resource("postgres-connections://list")
def get_postgres_connections_resource() -> Dict[str, Any]:
    return {"connections": [{"name": n, "status": "connected" if c.connected else "disconnected"} for n, c in db_connections.items()]}

if __name__ == "__main__":
    # For local testing, you might want to set GEMINI_API_KEY environment variable
    # Example: GEMINI_API_KEY="your_key" python postgres_server.py
    # The server will run on default host/port (e.g., localhost:8000 or as configured by FastMCP)
    print("Starting PostgreSQL MCP Server...")
    print("Ensure GEMINI_API_KEY environment variable is set if you want to use schema summarization with Gemini.")
    mcp.run()
