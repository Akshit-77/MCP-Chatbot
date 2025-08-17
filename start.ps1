# MCP Data Analysis Chatbot - Start Script
# This script starts the Streamlit application

param(
    [switch]$Debug,
    [string]$Port = "8501"
)

Write-Host "Starting MCP Data Analysis Chatbot..." -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "Virtual environment not found. Please run setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host ".env file not found. Please run setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Check if API key is configured
$envContent = Get-Content ".env" -Raw
if ($envContent -match "your_anthropic_api_key_here") {
    Write-Host "WARNING: Anthropic API key not configured!" -ForegroundColor Red
    Write-Host "Please edit .env file and add your API key before starting." -ForegroundColor Yellow
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Check if dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$streamlitCheck = uv run python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dependencies not installed. Installing now..." -ForegroundColor Yellow
    uv sync
}

# Start the Streamlit application
Write-Host "Starting Streamlit application on port $Port..." -ForegroundColor Green
Write-Host ""
Write-Host "Application will be available at: http://localhost:$Port" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

if ($Debug) {
    # Run in debug mode with verbose output
    uv run streamlit run client/app.py --server.port $Port --logger.level debug
} else {
    # Normal mode
    uv run streamlit run client/app.py --server.port $Port
}