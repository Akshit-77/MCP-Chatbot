# MCP Data Analysis Chatbot - Setup Script
# This script sets up the development environment for the MCP Data Analysis Chatbot

Write-Host "Setting up MCP Data Analysis Chatbot..." -ForegroundColor Green

# Check if UV is installed
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "UV package manager not found. Please install UV first." -ForegroundColor Red
    Write-Host "Visit: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
    exit 1
}

# Check if Python 3.12 is available
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = uv python find 3.12 2>$null
if (-not $pythonVersion) {
    Write-Host "Python 3.12 not found. Installing Python 3.12 with UV..." -ForegroundColor Yellow
    uv python install 3.12
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    uv venv --python 3.12
}

# Activate virtual environment and install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
uv sync

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.template" ".env"
    Write-Host "Please edit .env file and add your Anthropic API key." -ForegroundColor Cyan
    Write-Host "You can get an API key from: https://console.anthropic.com/" -ForegroundColor Cyan
}

# Check if .env file has API key configured
$envContent = Get-Content ".env" -Raw
if ($envContent -match "your_anthropic_api_key_here") {
    Write-Host "" -ForegroundColor Red
    Write-Host "WARNING: Please configure your Anthropic API key in the .env file!" -ForegroundColor Red
    Write-Host "Edit the .env file and replace 'your_anthropic_api_key_here' with your actual API key." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Configure your Anthropic API key in .env file" -ForegroundColor White
Write-Host "2. Run './start.ps1' to start the application" -ForegroundColor White
Write-Host "3. Open your browser to http://localhost:8501" -ForegroundColor White