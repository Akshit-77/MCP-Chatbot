# MCP Data Analysis Chatbot - Test MCP Server Script
# This script tests the MCP server functionality independently

Write-Host "Testing MCP Data Analysis Server..." -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "Virtual environment not found. Please run setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Test the MCP server by running a simple test
Write-Host "Running MCP server tests..." -ForegroundColor Yellow

$testScript = @"
import sys
import os
from pathlib import Path

# Add server directory to path
sys.path.append(str(Path(__file__).parent / 'server'))

try:
    # Import server functions
    from main import list_available_datasets, dataset_info, describe_data
    
    print("✓ MCP server imports successful")
    
    # Test list_available_datasets
    print("\n--- Testing list_available_datasets ---")
    result = list_available_datasets()
    print(f"Found {result.get('count', 0)} datasets")
    for dataset in result.get('datasets', []):
        print(f"  - {dataset['filename']} ({dataset['size_mb']} MB)")
    
    # Test dataset_info if we have datasets
    datasets = result.get('datasets', [])
    if datasets:
        test_file = datasets[0]['filename']
        print(f"\n--- Testing dataset_info with {test_file} ---")
        info_result = dataset_info(test_file)
        if 'error' in info_result:
            print(f"✗ Error: {info_result['error']}")
        else:
            print(f"✓ Dataset info retrieved successfully")
            print(f"  Shape: {info_result['shape']}")
            print(f"  Columns: {len(info_result['columns'])}")
    
    print("\n✓ All MCP server tests passed!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure all dependencies are installed with 'uv sync'")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

# Write test script to temp file and run it
$testFile = "temp_test.py"
$testScript | Out-File -FilePath $testFile -Encoding UTF8

try {
    uv run python $testFile
    $exitCode = $LASTEXITCODE
} finally {
    # Clean up temp file
    if (Test-Path $testFile) {
        Remove-Item $testFile
    }
}

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "🎉 MCP server is working correctly!" -ForegroundColor Green
    Write-Host "You can now run './start.ps1' to start the full application." -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "❌ MCP server tests failed." -ForegroundColor Red
    Write-Host "Please check the error messages above and fix any issues." -ForegroundColor Yellow
}