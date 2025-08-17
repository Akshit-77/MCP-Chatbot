#!/usr/bin/env python3
"""Test script to verify MCP tools are working properly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    # Test imports
    print("Testing imports...")
    from server.tools import list_available_datasets, dataset_info, load_dataset
    print("✓ Imports successful")
    
    # Test list_available_datasets
    print("\nTesting list_available_datasets...")
    datasets_result = list_available_datasets()
    print(f"Result: {datasets_result}")
    
    # Test dataset_info with iris.csv if available
    if datasets_result.get('count', 0) > 0:
        datasets = datasets_result.get('datasets', [])
        iris_dataset = None
        for dataset in datasets:
            if 'iris' in dataset['filename'].lower():
                iris_dataset = dataset['filename']
                break
        
        if iris_dataset:
            print(f"\nTesting dataset_info with {iris_dataset}...")
            info_result = dataset_info(iris_dataset)
            print(f"Dataset info result keys: {list(info_result.keys())}")
            if 'shape' in info_result:
                print(f"Dataset shape: {info_result['shape']}")
                print(f"Number of rows: {info_result['shape'][0]}")
                print(f"Number of columns/features: {info_result['shape'][1]}")
        else:
            print("Iris dataset not found")
    
    print("\n✓ All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()