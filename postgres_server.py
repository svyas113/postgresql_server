#!/usr/bin/env python3
"""
PostgreSQL Database MCP Server

This MCP server provides tools for:
1. Connecting to a PostgreSQL database using a connection string
2. Executing SQL queries on the connected database
3. Retrieving query results and schema information (including CREATE TABLE DDLs)
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

# MCP SDK
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("PostgreSQL Database Server")

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
            self.connection.autocommit = False # Explicitly manage transactions
            self.connected = True
            return True
        except Exception as e:
            print(f"Error connecting to PostgreSQL database '{self.db_name}': {e}")
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
                    WHERE schemaname = %s;
                """)
                cursor.execute(query, (schema_name,))
                tables = [row[0] for row in cursor.fetchall()]
                return tables
        except Exception as e:
            print(f"Error getting tables from PostgreSQL schema '{schema_name}' in db '{self.db_name}': {e}")
            return []

    def get_table_ddl(self, table_name: str, schema_name: str = 'public') -> Optional[str]:
        """Get the CREATE TABLE DDL statement for a specific table."""
        if not self.connected or not self.connection:
            return f"-- Not connected to database {self.db_name}"
        try:
            with self.connection.cursor() as cursor:
                # Attempt to use pg_get_tabledef for accurate DDL
                query_pg_get_tabledef = psycopg2.sql.SQL("""
                    SELECT pg_catalog.pg_get_tabledef(c.oid, true)
                    FROM pg_catalog.pg_class c
                    LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relkind = 'r' -- 'r' for ordinary table
                    AND c.relname = %s
                    AND n.nspname = %s;
                """)
                cursor.execute(query_pg_get_tabledef, (table_name, schema_name))
                result = cursor.fetchone()
                
                if result and result[0]:
                    return result[0]
                else:
                    # Fallback: Construct a basic CREATE TABLE from column info
                    print(f"pg_get_tabledef failed for {schema_name}.{table_name} in db '{self.db_name}'. Falling back to basic DDL.")
                    cols_info = self.get_table_column_info(table_name, schema_name)
                    if not cols_info:
                        return f"-- Could not retrieve DDL for {schema_name}.{table_name}"
                    
                    col_defs = []
                    for col in cols_info:
                        col_def = f"    {psycopg2.sql.Identifier(col['name']).as_string(self.connection)} {col['type']}"
                        if col.get('max_length'):
                             col_def += f"({col['max_length']})"
                        if not col['is_nullable']:
                            col_def += " NOT NULL"
                        if col['default_value']:
                            col_def += f" DEFAULT {col['default_value']}" # Default value might need more careful quoting
                        col_defs.append(col_def)
                    
                    return f"CREATE TABLE {psycopg2.sql.Identifier(schema_name).as_string(self.connection)}.{psycopg2.sql.Identifier(table_name).as_string(self.connection)} (\n" + ",\n".join(col_defs) + "\n);"

        except Exception as e:
            print(f"Error getting DDL for table {schema_name}.{table_name} in db '{self.db_name}': {e}")
            return f"-- Error retrieving DDL for {schema_name}.{table_name}: {e}"

    def get_table_column_info(self, table_name: str, schema_name: str = 'public') -> List[Dict[str, Any]]:
        """Get column information for a table."""
        if not self.connected or not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = psycopg2.sql.SQL("""
                    SELECT 
                        column_name, 
                        data_type, 
                        udt_name, -- underlying data type
                        character_maximum_length,
                        column_default,
                        is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = %s
                    ORDER BY ordinal_position;
                """)
                cursor.execute(query, (table_name, schema_name))
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        "name": row["column_name"],
                        "type": row["udt_name"], # More specific type
                        "data_type_standard": row["data_type"], # Standard SQL type
                        "max_length": row["character_maximum_length"],
                        "default_value": row["column_default"],
                        "is_nullable": row["is_nullable"] == "YES"
                    })
                return columns
        except Exception as e:
            print(f"Error getting column info for {schema_name}.{table_name} in db '{self.db_name}': {e}")
            return []
    
    def execute_query(self, query: str, params: Optional[List[Any]] = None, row_limit: Optional[int] = None) -> Dict[str, Any]:
        """Execute a SQL query and return the results."""
        if not self.connected or not self.connection:
            return {"error": f"Not connected to PostgreSQL database '{self.db_name}'"}
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                start_time = datetime.datetime.now()
                
                cursor.execute(query, params if params else None)
                
                if query.strip().lower().startswith("select"):
                    columns = [desc[0] for desc in cursor.description]
                    
                    if row_limit is not None:
                        rows_tuples = cursor.fetchmany(row_limit)
                    else:
                        rows_tuples = cursor.fetchall()
                    
                    rows = [dict(row) for row in rows_tuples]

                    end_time = datetime.datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    
                    return {
                        "columns": columns,
                        "rows": rows,
                        "row_count": len(rows),
                        "execution_time": execution_time
                    }
                else:
                    # For non-SELECT (INSERT, UPDATE, DELETE), commit changes
                    self.connection.commit()
                    end_time = datetime.datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    
                    return {
                        "affected_rows": cursor.rowcount,
                        "execution_time": execution_time,
                        "message": "Query executed and committed successfully."
                    }
        except Exception as e:
            if self.connection: # Rollback on error
                self.connection.rollback()
            return {"error": str(e)}
    
    def get_table_data(self, table_name: str, schema_name: str = 'public', limit: Optional[int] = None) -> pd.DataFrame:
        """Get data from a table as a pandas DataFrame."""
        if not self.connected or not self.connection:
            return pd.DataFrame()
        
        try:
            sql_query_base = psycopg2.sql.SQL("SELECT * FROM {}.{}").format(
                psycopg2.sql.Identifier(schema_name),
                psycopg2.sql.Identifier(table_name)
            )
            
            final_query_str: str
            params_tuple: Optional[tuple] = None

            if limit is not None:
                sql_query_limited = psycopg2.sql.SQL(" ").join([sql_query_base, psycopg2.sql.SQL("LIMIT %s")])
                final_query_str = sql_query_limited.as_string(self.connection)
                params_tuple = (limit,)
            else:
                final_query_str = sql_query_base.as_string(self.connection)

            return pd.read_sql_query(final_query_str, self.connection, params=params_tuple)
        except Exception as e:
            print(f"Error getting table data for {schema_name}.{table_name} in db '{self.db_name}': {e}")
            return pd.DataFrame()

#############################################################
# MCP Tools                                                 #
#############################################################

@mcp.tool()
def connect_to_postgres(connection_string: str, db_name: str) -> Dict[str, Any]:
    """
    Connect to a PostgreSQL database using a connection string.
    Example connection string: "postgresql://user:password@host:port/dbname"
    
    Args:
        connection_string: PostgreSQL connection string.
        db_name: A unique name to identify this database connection.
        
    Returns:
        Dictionary with connection result, including list of tables in 'public' schema.
    """
    if db_name in db_connections:
        return {
            "success": False,
            "message": f"Connection with name '{db_name}' already exists."
        }
    
    connection = PostgreSQLConnection(connection_string, db_name)
    if not connection.connect():
        return {
            "success": False,
            "message": f"Failed to connect to PostgreSQL database for '{db_name}'."
        }
    
    db_connections[db_name] = connection
    tables_public_schema = connection.get_tables(schema_name='public') # Default to public schema
    
    return {
        "success": True,
        "message": f"Connected to PostgreSQL database '{db_name}'. Found {len(tables_public_schema)} tables in 'public' schema.",
        "db_name": db_name,
        "public_schema_tables": tables_public_schema
    }

@mcp.tool()
def disconnect_postgres(db_name: str) -> Dict[str, Any]:
    """
    Disconnect from a PostgreSQL database.
    
    Args:
        db_name: Name of the database connection to disconnect.
        
    Returns:
        Dictionary with disconnection result.
    """
    if db_name not in db_connections:
        return {
            "success": False,
            "message": f"No PostgreSQL connection found with name '{db_name}'."
        }
    
    connection = db_connections[db_name]
    if connection.disconnect():
        del db_connections[db_name]
        return {
            "success": True,
            "message": f"Disconnected from PostgreSQL database '{db_name}'."
        }
    else:
        return {
            "success": False,
            "message": f"Failed to disconnect from PostgreSQL database '{db_name}'."
        }

@mcp.tool()
def list_postgres_connections() -> Dict[str, Any]:
    """
    List all active PostgreSQL database connections.
    
    Returns:
        Dictionary with list of active database connections.
    """
    return {
        "connections": [
            {
                "name": name,
                # Avoid exposing full connection string. Could parse and show host/db if needed.
                "status": "connected" if conn.connected else "disconnected" 
            } for name, conn in db_connections.items()
        ]
    }

@mcp.tool()
def execute_postgres_query(db_name: str, query: str, params: Optional[List[Any]] = None, row_limit: int = 100) -> Dict[str, Any]:
    """
    Execute SQL query on a PostgreSQL database.
    
    Args:
        db_name: Name of the database connection.
        query: SQL query to execute.
        params: Optional list of parameters for the query (using %s placeholders).
        row_limit: Maximum number of rows to return for SELECT queries (default 100).
        
    Returns:
        Dictionary with query results or error message.
    """
    if db_name not in db_connections:
        return {
            "success": False,
            "message": f"No PostgreSQL connection found with name '{db_name}'."
        }
    
    connection = db_connections[db_name]
    # Pass row_limit only if it's a SELECT query, execute_query handles this.
    actual_row_limit = row_limit if query.strip().lower().startswith("select") else None
    result = connection.execute_query(query, params=params, row_limit=actual_row_limit)
    
    if "error" in result:
        return {
            "success": False,
            "message": result["error"]
        }
    
    formatted_result = {
        "success": True,
        "result": result
    }
    
    if query.strip().lower().startswith("select") and "columns" in result and "rows" in result:
        try:
            df = pd.DataFrame(result["rows"], columns=result["columns"])
            formatted_result["formatted_table"] = df.to_markdown(index=False)
        except Exception as e:
            print(f"Error formatting table for db '{db_name}': {e}")
            formatted_result["formatted_table_error"] = str(e)
    
    return formatted_result

@mcp.tool()
def get_postgres_table_info(db_name: str, table_name: str, schema_name: str = 'public') -> Dict[str, Any]:
    """
    Get detailed column information for a specific table in a PostgreSQL database.
    
    Args:
        db_name: Name of the database connection.
        table_name: Name of the table.
        schema_name: Name of the schema (default 'public').
        
    Returns:
        Dictionary with table column information and sample data.
    """
    if db_name not in db_connections:
        return {"success": False, "message": f"No PostgreSQL connection found with name '{db_name}'."}
    
    connection = db_connections[db_name]
    columns = connection.get_table_column_info(table_name, schema_name)
    
    if not columns:
        return {
            "success": False,
            "message": f"Error getting column information for table '{schema_name}.{table_name}' in db '{db_name}'."
        }
    
    sample_data_query = psycopg2.sql.SQL("SELECT * FROM {}.{} LIMIT 5").format(
        psycopg2.sql.Identifier(schema_name),
        psycopg2.sql.Identifier(table_name)
    ).as_string(connection.connection) # Get string representation for execute_query

    sample_data_result = connection.execute_query(sample_data_query)
    
    return {
        "success": True,
        "db_name": db_name,
        "schema_name": schema_name,
        "table_name": table_name,
        "columns": columns,
        "sample_data": sample_data_result if "error" not in sample_data_result else {"error": sample_data_result.get("error")}
    }

@mcp.tool()
def get_postgres_table_data(db_name: str, table_name: str, schema_name: str = 'public', limit: int = 100) -> Dict[str, Any]:
    """
    Get data from a specific table in a PostgreSQL database.
    
    Args:
        db_name: Name of the database connection.
        table_name: Name of the table.
        schema_name: Name of the schema (default 'public').
        limit: Maximum number of rows to return (default 100).
        
    Returns:
        Dictionary with table data.
    """
    if db_name not in db_connections:
        return {"success": False, "message": f"No PostgreSQL connection found with name '{db_name}'."}
    
    connection = db_connections[db_name]
    
    # Construct the query string for execute_query
    query_base = psycopg2.sql.SQL("SELECT * FROM {}.{}").format(
        psycopg2.sql.Identifier(schema_name),
        psycopg2.sql.Identifier(table_name)
    )
    query_final = psycopg2.sql.SQL(" ").join([query_base, psycopg2.sql.SQL("LIMIT %s")])
    query_str = query_final.as_string(connection.connection)

    result = connection.execute_query(query_str, params=[limit]) # params must be a list or tuple
    
    if "error" in result:
        return {"success": False, "message": result["error"]}
    
    formatted_result = {
        "success": True,
        "db_name": db_name,
        "schema_name": schema_name,
        "table_name": table_name,
        "result": result
    }
    
    try:
        if "columns" in result and "rows" in result:
            df = pd.DataFrame(result["rows"], columns=result["columns"])
            formatted_result["formatted_table"] = df.to_markdown(index=False)
    except Exception as e:
        print(f"Error formatting table for {schema_name}.{table_name} in db '{db_name}': {e}")
        formatted_result["formatted_table_error"] = str(e)
            
    return formatted_result

#############################################################
# MCP Resources                                             #
#############################################################

@mcp.resource("postgres-schema://{db_name}/{schema_name}")
def get_postgres_schema_resource(db_name: str, schema_name: str) -> Dict[str, Any]:
    """
    Get PostgreSQL database schema information (table DDLs) for a specific schema.
    
    Args:
        db_name: Name of the database connection.
        schema_name: Name of the schema to retrieve DDLs for.
    """
    if db_name not in db_connections:
        return {"error": f"No PostgreSQL connection found with name '{db_name}'."}
    
    connection = db_connections[db_name]
    if not connection.connected:
         return {"error": f"PostgreSQL connection '{db_name}' is not active."}

    tables = connection.get_tables(schema_name)
    schema_ddls: Dict[str, Optional[str]] = {}
    
    for table in tables:
        ddl = connection.get_table_ddl(table, schema_name)
        schema_ddls[table] = ddl
    
    return {
        "db_name": db_name,
        "schema_name": schema_name,
        "table_ddls": schema_ddls
    }

@mcp.resource("postgres-table://{db_name}/{schema_name}/{table_name}")
def get_postgres_table_resource(db_name: str, schema_name: str, table_name: str) -> Dict[str, Any]:
    """
    Get PostgreSQL table information (columns and sample data).
    
    Args:
        db_name: Name of the database connection.
        schema_name: Name of the schema.
        table_name: Name of the table.
    """
    if db_name not in db_connections:
        return {"error": f"No PostgreSQL connection found with name '{db_name}'."}
    
    connection = db_connections[db_name]
    if not connection.connected:
         return {"error": f"PostgreSQL connection '{db_name}' is not active."}

    columns = connection.get_table_column_info(table_name, schema_name)
    
    sample_data_query = psycopg2.sql.SQL("SELECT * FROM {}.{} LIMIT 5").format(
        psycopg2.sql.Identifier(schema_name),
        psycopg2.sql.Identifier(table_name)
    ).as_string(connection.connection)
    sample_data_result = connection.execute_query(sample_data_query)
    
    return {
        "db_name": db_name,
        "schema_name": schema_name,
        "table_name": table_name,
        "columns": columns,
        "sample_data": sample_data_result if "error" not in sample_data_result else {"error": sample_data_result.get("error")}
    }

@mcp.resource("postgres-connections://list")
def get_postgres_connections_resource() -> Dict[str, Any]:
    """Get list of active PostgreSQL database connections."""
    return {
        "connections": [
            {
                "name": name,
                "status": "connected" if conn.connected else "disconnected"
            } for name, conn in db_connections.items()
        ]
    }

# Run the MCP server
if __name__ == "__main__":
    # Example: mcp.run(host="0.0.0.0", port=8001)
    mcp.run()
