# PostgreSQL Database MCP Server

This MCP (Model Context Protocol) server provides tools and resources to interact with PostgreSQL databases. It allows Cline to connect to databases, execute queries, retrieve schema information, and fetch table data.

## Features

### Tools

The server exposes the following tools for use with Cline:

*   **`connect_to_postgres(connection_string: str, db_name: str)`**:
    *   Establishes a connection to a PostgreSQL database using a standard connection string (e.g., `"postgresql://user:password@host:port/dbname"`).
    *   Assigns a unique `db_name` to this connection for future reference.
    *   Returns a list of tables found in the schema upon successful connection.

*   **`disconnect_postgres(db_name: str)`**:
    *   Closes the connection to the specified PostgreSQL database.

*   **`list_postgres_connections()`**:
    *   Lists all currently active PostgreSQL database connections managed by the server, showing their assigned `db_name` and status.

*   **`execute_postgres_query(db_name: str, query: str, params: Optional[List[Any]] = None, row_limit: int = 100)`**:
    *   Executes an SQL query on the specified database connection.
    *   Supports parameterized queries (using `%s` placeholders and a list of `params`).
    *   For `SELECT` queries, returns column names, rows (up to `row_limit`), row count, execution time, and a Markdown-formatted table of the results.
    *   For other queries (INSERT, UPDATE, DELETE), it commits the transaction and returns the number of affected rows and execution time.

*   **`get_postgres_table_info(db_name: str, table_name: str, schema_name: str = 'public')`**:
    *   Retrieves detailed information about a specific table, including:
        *   Column names, data types (actual and standard), max length, default values, and nullability.
        *   A sample of up to 5 rows from the table.

*   **`get_postgres_table_data(db_name: str, table_name: str, schema_name: str = 'public', limit: int = 100)`**:
    *   Fetches data from a specified table.
    *   Returns column names, rows (up to `limit`), and a Markdown-formatted table of the results.

### Resources

The server provides the following resources that Cline can access:

*   **`postgres-schema://{db_name}/{schema_name}`**:
    *   Provides the Data Definition Language (DDL) statements (e.g., `CREATE TABLE ...`) for all tables within the specified `schema_name` of the connected `db_name`.

*   **`postgres-table://{db_name}/{schema_name}/{table_name}`**:
    *   Provides detailed information for a specific `table_name` within a `schema_name` of the connected `db_name`. This includes column details and a sample of 5 rows from the table.

*   **`postgres-connections://list`**:
    *   Provides a list of all active PostgreSQL database connections, including their names and statuses.

## Requirements

To use this MCP server, you need:

*   Python 3.13
*   The Python packages listed in `postgres_requirements.txt`:
    *   `psycopg2-binary` (for PostgreSQL database adapter)
    *   `pandas` (for data handling and Markdown table formatting)
    *   `mcp-sdk` (for running the MCP server)

You can install these using pip:
```bash
pip install -r postgres_requirements.txt
```

## Setup

1.  **Clone the Repository**:
    If this server is part of a larger repository, clone that repository.
    ```bash
    # Example: git clone https://github.com/your-username/your-repo.git
    # cd your-repo/mcp_servers
    ```

2.  **Install Dependencies**:
    Navigate to the directory containing `postgres_server.py` and `postgres_requirements.txt` (e.g., `mcp_servers/`) and install the required Python packages:
    ```bash
    pip install -r postgres_requirements.txt
    ```

3.  **Ensure PostgreSQL is Accessible**:
    The machine running this MCP server must have network access to the PostgreSQL database(s) you intend to connect to.

## Usage

### Connecting with Cline

Once all the dependencies are installed, you need to configure Cline to connect to it. This is typically done by editing Cline's MCP settings/configuration file.

**Locating Cline's MCP Settings File:**

On Windows, this file is usually located at:
`C:\Users\<YourUserName>\AppData\Roaming\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

(Replace `<YourUserName>` with your actual Windows username.)

**Adding the Server to `cline_mcp_settings.json`:**

You'll need to add an entry for this "PostgreSQL Database Server" to the `mcpServers` object in the JSON file. The server name defined in `postgres_server.py` is "PostgreSQL Database Server".

Here's a generic example of how you might add a *new* MCP server entry. Adapt this structure for the "PostgreSQL Database Server":

```json
{
  "mcpServers": {
    // ... other existing server configurations ...

    "YourNewServerName": {
      "command": "path/to/your/python_or_executable", // e.g., "C:/path/to/.venv/Scripts/python.exe" or "node"
      "args": ["path/to/your/server_script.py"],    // e.g., ["C:/path/to/mcp_servers/your_server_script.py"]
      "env": {}, // Optional: environment variables for the server process
      "disabled": false, // Set to true to temporarily disable the server
      "autoApprove": [] // Optional: list of tool names from this server to auto-approve
    },

    // Specifically for this PostgreSQL server, it would look something like:
    "PostgreSQL Database Server": {
      "command": "C:/path/to/your/python_env/Scripts/python.exe", // Adjust to your Python interpreter
      "args": ["C:/path/to/your_clone/mcp_servers/postgres_server.py"], // Adjust to the actual path of postgres_server.py
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

**Important Notes for Configuration:**
*   Replace `"C:/path/to/your/python_env/Scripts/python.exe"` with the actual path to the Python interpreter you want to use to run the server (ideally from the virtual environment where you installed the dependencies).
*   Replace `"C:/path/to/your_clone/mcp_servers/postgres_server.py"` with the actual absolute path to the `postgres_server.py` script on your system.
*   The key ("PostgreSQL Database Server" in the example above) must match the `name` argument passed to `FastMCP()` in the server script if you want Cline to automatically identify it by that name. However, Cline uses the key in the JSON as the `server_name` for `use_mcp_tool` and `access_mcp_resource`. It's good practice to keep these consistent.

After saving changes to `cline_mcp_settings.json`, you might need to restart Cline your VS Code for the changes to take effect. Once Cline is configured and connected to this MCP server, you can use the tools and access the resources listed above.

**Example Cline Interaction (Conceptual):**

```
User: Connect to my sales database. Connection string is "postgresql://user:pass@localhost:5432/sales_db" and let's call it "my_sales_db".

Cline (using connect_to_postgres tool):
<use_mcp_tool>
  <server_name>PostgreSQL Database Server</server_name>
  <tool_name>connect_to_postgres</tool_name>
  <arguments>
    {
      "connection_string": "postgresql://user:pass@localhost:5432/sales_db",
      "db_name": "my_sales_db"
    }
  </arguments>
</use_mcp_tool>

User: Show me the schema for the 'public' schema in 'my_sales_db'.

Cline (accessing postgres-schema resource):
<access_mcp_resource>
  <server_name>PostgreSQL Database Server</server_name>
  <uri>postgres-schema://my_sales_db/public</uri>
</access_mcp_resource>

User: Execute a query on 'my_sales_db': "SELECT * FROM customers LIMIT 10;"

Cline (using execute_postgres_query tool):
<use_mcp_tool>
  <server_name>PostgreSQL Database Server</server_name>
  <tool_name>execute_postgres_query</tool_name>
  <arguments>
    {
      "db_name": "my_sales_db",
      "query": "SELECT * FROM customers LIMIT 10;",
      "row_limit": 10
    }
  </arguments>
</use_mcp_tool>
```

This MCP server enhances Cline's capabilities by allowing it to directly interact with PostgreSQL databases, making it a powerful assistant for database-related tasks.
