# 📊 MCP Data Analysis Chatbot

A powerful data analysis chatbot built with Streamlit and Claude AI, featuring MCP (Model Context Protocol) integration for seamless dataset analysis through natural language queries.

## ✨ Features

- **Natural Language Data Analysis**: Ask questions about your datasets in plain English
- **MCP Server Integration**: Custom MCP server with comprehensive data analysis tools
- **Interactive Visualizations**: Automatic generation of charts, plots, and statistical visualizations
- **Multiple File Formats**: Support for CSV, Excel (.xlsx, .xls) files
- **Session Management**: Save and restore chat sessions with persistent context
- **Dataset Upload**: Easy file upload with validation and management
- **Real-time Processing**: Fast analysis powered by pandas, matplotlib, and seaborn
- **Windows Optimized**: Designed specifically for Windows environments with PowerShell automation

## 🎯 Use Cases

- **Business Analytics**: Analyze sales data, customer metrics, and performance indicators
- **Data Exploration**: Quick insights and statistical summaries of datasets
- **Visualization Generation**: Create charts and graphs with simple text commands
- **Data Cleaning**: Handle missing values, duplicates, and data quality issues
- **Educational Research**: Learn about datasets through interactive exploration

## 🛠️ Prerequisites

- **Windows 10/11** with PowerShell
- **Python 3.12** (will be installed automatically via UV)
- **UV Package Manager** - [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Anthropic API Key** - [Get one here](https://console.anthropic.com/)
- **VS Code** (recommended for development)

## 🚀 Quick Start

### 1. Clone and Setup

```powershell
# Clone the repository (or extract if you have the files)
cd mcp-data-chatbot

# Run the setup script
./setup.ps1
```

### 2. Configure API Key

Edit the `.env` file and add your Anthropic API key:

```env
ANTHROPIC_API_KEY=your_actual_api_key_here
```

### 3. Test the Installation

```powershell
# Test MCP server functionality
./test-mcp.ps1
```

### 4. Start the Application

```powershell
# Start the Streamlit app
./start.ps1

# Or with debug mode
./start.ps1 -Debug

# Or on a different port
./start.ps1 -Port 8502
```

### 5. Open Your Browser

Navigate to `http://localhost:8501` and start analyzing your data!

## 📁 Project Structure

```
mcp-data-chatbot/
├── server/              # MCP server implementation
│   └── main.py         # Core MCP tools and data analysis functions
├── client/             # Streamlit frontend
│   ├── app.py          # Main Streamlit application
│   └── mcp_client.py   # MCP client integration
├── datasets/           # Sample and uploaded datasets
│   ├── iris.csv        # Classic iris dataset
│   ├── sales_data.csv  # Business sales example
│   └── weather.csv     # Weather time series data
├── sessions/           # Saved chat sessions (auto-created)
├── .env.template       # Environment variables template
├── pyproject.toml      # Python dependencies
├── setup.ps1           # Setup script
├── start.ps1           # Application startup script
├── test-mcp.ps1        # MCP server testing script
└── README.md           # This file
```

## 🔧 Available Analysis Tools

The MCP server provides these data analysis capabilities:

### Dataset Management
- **`list_available_datasets()`** - List all available datasets
- **`load_dataset(filename)`** - Load and validate a dataset
- **`dataset_info(filename)`** - Get comprehensive dataset information

### Statistical Analysis
- **`describe_data(filename)`** - Generate descriptive statistics
- **`correlation_analysis(filename)`** - Create correlation matrices and heatmaps
- **`group_analysis(filename, group_by_column)`** - Perform group-by operations

### Data Visualization
- **`simple_plot(filename, plot_type, x_column, y_column)`** - Generate various plot types:
  - Histograms
  - Scatter plots
  - Bar charts
  - Line plots
  - Box plots

### Data Processing
- **`clean_data(filename)`** - Handle missing values and duplicates
- **`filter_data(filename, conditions)`** - Filter datasets with conditions
- **`sample_data(filename, n_rows)`** - Create random samples

## 💬 Example Queries

Try these natural language queries with your datasets:

```
"Show me basic statistics for the iris dataset"
"Create a scatter plot of sepal length vs sepal width for iris.csv"
"What are the top selling products in sales_data.csv?"
"Generate a correlation heatmap for the numeric columns"
"Clean the dataset and remove missing values"
"Filter the sales data for Electronics category only"
"Group the weather data by city and show average temperature"
"Create a histogram of sales amounts"
```

## 🎨 Sample Datasets

The project includes three sample datasets to get you started:

1. **iris.csv** - Classic machine learning dataset with flower measurements
2. **sales_data.csv** - Business sales data with products, regions, and salespeople
3. **weather.csv** - Weather data for multiple cities with temperature, humidity, etc.

## 🔧 Configuration Options

### Environment Variables (.env)

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional
CLAUDE_MODEL=claude-3-5-sonnet-20241022
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost
MAX_FILE_SIZE_MB=100
ALLOWED_FILE_TYPES=csv,xlsx,xls
SESSION_TIMEOUT_HOURS=24
MAX_SESSIONS=100
```

### Streamlit Configuration

The app automatically configures Streamlit with optimal settings for data analysis:
- Wide layout for better visualization
- Expanded sidebar for easy access to tools
- Custom theming for professional appearance

## 🚨 Troubleshooting

### Common Issues

**1. "UV not found" error**
- Install UV package manager: `https://docs.astral.sh/uv/getting-started/installation/`

**2. "Python 3.12 not found" error**
- UV will automatically install Python 3.12, but you may need to restart your terminal

**3. "Anthropic API key not configured" error**
- Make sure you've edited the `.env` file with your actual API key
- Verify the API key is valid at https://console.anthropic.com/

**4. Import errors or missing dependencies**
- Run `uv sync` to ensure all dependencies are installed
- Try `./setup.ps1` again

**5. Streamlit won't start**
- Check if port 8501 is already in use
- Try starting with a different port: `./start.ps1 -Port 8502`

**6. Dataset upload issues**
- Ensure file size is under 100MB
- Verify file format is CSV or Excel
- Check file isn't corrupted or password-protected

### Performance Tips

- **Large Datasets**: For files over 10MB, consider using the sampling tool first
- **Memory Usage**: Close browser tabs and restart the app if memory usage is high
- **Visualization**: Complex plots may take longer to generate for large datasets

### Getting Help

1. Run `./test-mcp.ps1` to diagnose MCP server issues
2. Check the Streamlit logs in the terminal for detailed error messages
3. Verify your dataset format matches the expected structure
4. Ensure all environment variables are properly configured

## 🔒 Security & Privacy

- **Local Processing**: All data analysis happens locally on your machine
- **No Data Upload**: Your datasets never leave your computer
- **API Security**: Only analysis instructions are sent to Claude API, not your raw data
- **Session Storage**: Chat sessions are stored locally in JSON format

## 🚀 Advanced Usage

### Custom Datasets

You can upload any CSV or Excel file by:
1. Using the file uploader in the sidebar
2. Placing files directly in the `datasets/` folder
3. Ensuring proper column headers and data formatting

### Extending the MCP Server

To add new analysis tools:
1. Add new functions to `server/main.py`
2. Decorate with `@mcp.tool()`
3. Update the tool definitions in `client/mcp_client.py`
4. Restart the application

### Session Management

- Sessions are automatically saved after each interaction
- Use "Load Session" to continue previous analysis
- Sessions include full conversation history and context
- Clear old sessions from the `sessions/` folder if needed

## 📈 Development

### Running in Development Mode

```powershell
# Debug mode with verbose logging
./start.ps1 -Debug

# Manual development setup
uv run streamlit run client/app.py --server.port 8501 --logger.level debug
```

### Code Structure

- **MCP Server** (`server/main.py`): FastMCP-based server with data analysis tools
- **MCP Client** (`client/mcp_client.py`): Handles communication between Streamlit and server
- **Streamlit App** (`client/app.py`): Main user interface and session management

## 📝 License

This project is provided as-is for educational and research purposes. Please respect the terms of service for:
- Anthropic Claude API
- Third-party libraries used in this project

## 🙋‍♂️ Support

For questions and support:
1. Check this README for common solutions
2. Run the diagnostic scripts (`test-mcp.ps1`)
3. Review the error messages in the terminal
4. Ensure all prerequisites are properly installed

---

**Happy Data Analysis! 📊✨**

*Built with ❤️ using Streamlit, Claude AI, and MCP*