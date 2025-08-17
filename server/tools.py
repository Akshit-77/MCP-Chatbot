#!/usr/bin/env python3
"""
Data Analysis Tools - Callable Functions

This module contains the actual callable functions that can be imported
and used directly, separate from the MCP tool decorators.
"""

import os
import sys
import json
import base64
import io
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import matplotlib
# Set matplotlib backend to Agg to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

def get_dataset_path(filename: str) -> Path:
    """Get the full path to a dataset file."""
    return DATASETS_DIR / filename

def validate_filename(filename: str) -> bool:
    """Validate that the filename has a supported extension."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS

def load_dataframe(filename: str) -> pd.DataFrame:
    """Load a dataset file into a pandas DataFrame."""
    if not validate_filename(filename):
        raise ValueError(f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    
    filepath = get_dataset_path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset '{filename}' not found in datasets directory")
    
    try:
        if filepath.suffix.lower() == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset '{filename}': {str(e)}")

def plot_to_base64(plt_figure) -> str:
    """Convert a matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    plt_figure.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plot_data = buffer.getvalue()
    buffer.close()
    plt.close(plt_figure)
    
    return base64.b64encode(plot_data).decode('utf-8')

# Dataset Management Tools

def list_available_datasets() -> Dict[str, Any]:
    """List all available datasets in the datasets directory.
    
    Returns:
        Dict containing list of available datasets with basic info
    """
    try:
        datasets = []
        for file_path in DATASETS_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                # Get basic file info
                stat = file_path.stat()
                datasets.append({
                    'filename': file_path.name,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'type': file_path.suffix.lower(),
                    'modified': stat.st_mtime
                })
        
        return {
            'datasets': datasets,
            'count': len(datasets),
            'supported_types': list(SUPPORTED_EXTENSIONS)
        }
    except Exception as e:
        return {'error': f"Error listing datasets: {str(e)}"}

def load_dataset(filename: str) -> Dict[str, Any]:
    """Load a dataset and return basic information about it.
    
    Args:
        filename: Name of the dataset file to load
        
    Returns:
        Dict containing dataset loading status and basic info
    """
    try:
        df = load_dataframe(filename)
        
        return {
            'success': True,
            'filename': filename,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            'sample_data': df.head(5).to_dict('records')
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def dataset_info(filename: str) -> Dict[str, Any]:
    """Get comprehensive information about a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing detailed dataset metadata and statistics
    """
    try:
        df = load_dataframe(filename)
        
        # Basic info
        info = {
            'filename': filename,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        
        # Missing values
        missing_data = df.isnull().sum()
        info['missing_values'] = missing_data[missing_data > 0].to_dict()
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_columns'] = numeric_cols.tolist()
            info['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['categorical_columns'] = categorical_cols.tolist()
            info['categorical_summary'] = {}
            for col in categorical_cols:
                unique_count = df[col].nunique()
                info['categorical_summary'][col] = {
                    'unique_values': unique_count,
                    'top_values': df[col].value_counts().head(5).to_dict() if unique_count <= 1000 else "Too many unique values"
                }
        
        # Sample data
        info['sample_data'] = df.head(10).to_dict('records')
        
        return info
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def create_feature_histograms(filename: str) -> Dict[str, Any]:
    """Create histograms for all numeric features in a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing histogram plots for all numeric features
    """
    try:
        df = load_dataframe(filename)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric columns found for histogram creation',
                'columns': df.columns.tolist()
            }
        
        # Calculate subplot dimensions
        n_cols = min(3, len(numeric_cols))  # Max 3 columns
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle single subplot case
        if len(numeric_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        # Create histograms
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Histogram of {col}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_base64 = plot_to_base64(fig)
        
        # Calculate basic statistics for each feature
        feature_stats = {}
        for col in numeric_cols:
            feature_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
        
        return {
            'filename': filename,
            'numeric_features': numeric_cols.tolist(),
            'plot_base64': plot_base64,
            'feature_statistics': feature_stats,
            'summary': f'Created histograms for {len(numeric_cols)} numeric features'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

# Analysis Tools

def describe_data(filename: str) -> Dict[str, Any]:
    """Generate descriptive statistics for a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing descriptive statistics
    """
    try:
        df = load_dataframe(filename)
        
        result = {
            'filename': filename,
            'basic_info': {
                'shape': df.shape,
                'total_cells': df.shape[0] * df.shape[1],
                'missing_cells': df.isnull().sum().sum(),
                'missing_percentage': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
            }
        }
        
        # Numeric columns description
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            result['numeric_description'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns description
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_desc = {}
            for col in categorical_cols:
                cat_desc[col] = {
                    'count': df[col].count(),
                    'unique': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'frequency_of_most': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
                }
            result['categorical_description'] = cat_desc
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def create_feature_histograms(filename: str) -> Dict[str, Any]:
    """Create histograms for all numeric features in a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing histogram plots for all numeric features
    """
    try:
        df = load_dataframe(filename)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric columns found for histogram creation',
                'columns': df.columns.tolist()
            }
        
        # Calculate subplot dimensions
        n_cols = min(3, len(numeric_cols))  # Max 3 columns
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle single subplot case
        if len(numeric_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        # Create histograms
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Histogram of {col}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_base64 = plot_to_base64(fig)
        
        # Calculate basic statistics for each feature
        feature_stats = {}
        for col in numeric_cols:
            feature_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
        
        return {
            'filename': filename,
            'numeric_features': numeric_cols.tolist(),
            'plot_base64': plot_base64,
            'feature_statistics': feature_stats,
            'summary': f'Created histograms for {len(numeric_cols)} numeric features'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def correlation_analysis(filename: str) -> Dict[str, Any]:
    """Perform correlation analysis on numeric columns.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing correlation matrix and analysis
    """
    try:
        df = load_dataframe(filename)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                'error': 'Need at least 2 numeric columns for correlation analysis',
                'numeric_columns_found': numeric_cols.tolist()
            }
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title(f'Correlation Matrix - {filename}')
        plt.tight_layout()
        
        # Convert plot to base64
        plot_base64 = plot_to_base64(plt.gcf())
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        
        return {
            'filename': filename,
            'correlation_matrix': corr_matrix.round(3).to_dict(),
            'strong_correlations': strong_correlations,
            'plot_base64': plot_base64,
            'numeric_columns': numeric_cols.tolist()
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def create_feature_histograms(filename: str) -> Dict[str, Any]:
    """Create histograms for all numeric features in a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing histogram plots for all numeric features
    """
    try:
        df = load_dataframe(filename)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric columns found for histogram creation',
                'columns': df.columns.tolist()
            }
        
        # Calculate subplot dimensions
        n_cols = min(3, len(numeric_cols))  # Max 3 columns
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle single subplot case
        if len(numeric_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        # Create histograms
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Histogram of {col}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_base64 = plot_to_base64(fig)
        
        # Calculate basic statistics for each feature
        feature_stats = {}
        for col in numeric_cols:
            feature_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
        
        return {
            'filename': filename,
            'numeric_features': numeric_cols.tolist(),
            'plot_base64': plot_base64,
            'feature_statistics': feature_stats,
            'summary': f'Created histograms for {len(numeric_cols)} numeric features'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def group_analysis(filename: str, group_by_column: str) -> Dict[str, Any]:
    """Perform group-by analysis on a dataset.
    
    Args:
        filename: Name of the dataset file
        group_by_column: Column name to group by
        
    Returns:
        Dict containing grouped analysis results
    """
    try:
        df = load_dataframe(filename)
        
        if group_by_column not in df.columns:
            return {
                'error': f"Column '{group_by_column}' not found in dataset",
                'available_columns': df.columns.tolist()
            }
        
        # Group by the specified column
        grouped = df.groupby(group_by_column)
        
        result = {
            'filename': filename,
            'group_by_column': group_by_column,
            'number_of_groups': grouped.ngroups,
            'group_sizes': grouped.size().to_dict()
        }
        
        # Numeric columns aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            agg_stats = grouped[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
            result['numeric_aggregations'] = {}
            for col in numeric_cols:
                result['numeric_aggregations'][col] = agg_stats[col].round(3).to_dict()
        
        # Create visualization
        if len(result['group_sizes']) <= 20:  # Only plot if reasonable number of groups
            plt.figure(figsize=(12, 6))
            
            # Group sizes plot
            plt.subplot(1, 2, 1)
            group_sizes = grouped.size()
            group_sizes.plot(kind='bar')
            plt.title(f'Group Sizes by {group_by_column}')
            plt.xlabel(group_by_column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # If there's at least one numeric column, plot its mean by group
            if len(numeric_cols) > 0:
                plt.subplot(1, 2, 2)
                first_numeric = numeric_cols[0]
                grouped[first_numeric].mean().plot(kind='bar')
                plt.title(f'Mean {first_numeric} by {group_by_column}')
                plt.xlabel(group_by_column)
                plt.ylabel(f'Mean {first_numeric}')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            result['plot_base64'] = plot_to_base64(plt.gcf())
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def create_feature_histograms(filename: str) -> Dict[str, Any]:
    """Create histograms for all numeric features in a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing histogram plots for all numeric features
    """
    try:
        df = load_dataframe(filename)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric columns found for histogram creation',
                'columns': df.columns.tolist()
            }
        
        # Calculate subplot dimensions
        n_cols = min(3, len(numeric_cols))  # Max 3 columns
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle single subplot case
        if len(numeric_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        # Create histograms
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Histogram of {col}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_base64 = plot_to_base64(fig)
        
        # Calculate basic statistics for each feature
        feature_stats = {}
        for col in numeric_cols:
            feature_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
        
        return {
            'filename': filename,
            'numeric_features': numeric_cols.tolist(),
            'plot_base64': plot_base64,
            'feature_statistics': feature_stats,
            'summary': f'Created histograms for {len(numeric_cols)} numeric features'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def simple_plot(filename: str, plot_type: str, x_column: str, y_column: Optional[str] = None) -> Dict[str, Any]:
    """Generate simple plots for data visualization.
    
    Args:
        filename: Name of the dataset file
        plot_type: Type of plot ('histogram', 'scatter', 'bar', 'line', 'box')
        x_column: Column for x-axis
        y_column: Column for y-axis (optional for some plot types)
        
    Returns:
        Dict containing plot information and base64 encoded image
    """
    try:
        df = load_dataframe(filename)
        
        if x_column not in df.columns:
            return {
                'error': f"Column '{x_column}' not found in dataset",
                'available_columns': df.columns.tolist()
            }
        
        if y_column and y_column not in df.columns:
            return {
                'error': f"Column '{y_column}' not found in dataset",
                'available_columns': df.columns.tolist()
            }
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'histogram':
            plt.hist(df[x_column].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel(x_column)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {x_column}')
            
        elif plot_type == 'scatter' and y_column:
            plt.scatter(df[x_column], df[y_column], alpha=0.6)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f'Scatter Plot: {x_column} vs {y_column}')
            
        elif plot_type == 'bar':
            if df[x_column].dtype in ['object', 'category']:
                value_counts = df[x_column].value_counts().head(20)
                value_counts.plot(kind='bar')
                plt.xlabel(x_column)
                plt.ylabel('Count')
                plt.title(f'Bar Plot of {x_column}')
                plt.xticks(rotation=45)
            else:
                return {'error': f'Bar plot requires categorical column, but {x_column} is numeric'}
                
        elif plot_type == 'line' and y_column:
            df_sorted = df.sort_values(x_column)
            plt.plot(df_sorted[x_column], df_sorted[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f'Line Plot: {x_column} vs {y_column}')
            
        elif plot_type == 'box':
            if y_column:
                # Box plot by category
                df.boxplot(column=y_column, by=x_column, figsize=(10, 6))
                plt.title(f'Box Plot of {y_column} by {x_column}')
            else:
                # Single box plot
                plt.boxplot(df[x_column].dropna())
                plt.ylabel(x_column)
                plt.title(f'Box Plot of {x_column}')
        else:
            return {
                'error': f"Unsupported plot type '{plot_type}' or missing required parameters",
                'supported_types': ['histogram', 'scatter', 'bar', 'line', 'box']
            }
        
        plt.tight_layout()
        plot_base64 = plot_to_base64(plt.gcf())
        
        return {
            'filename': filename,
            'plot_type': plot_type,
            'x_column': x_column,
            'y_column': y_column,
            'plot_base64': plot_base64
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def create_feature_histograms(filename: str) -> Dict[str, Any]:
    """Create histograms for all numeric features in a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing histogram plots for all numeric features
    """
    try:
        df = load_dataframe(filename)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric columns found for histogram creation',
                'columns': df.columns.tolist()
            }
        
        # Calculate subplot dimensions
        n_cols = min(3, len(numeric_cols))  # Max 3 columns
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle single subplot case
        if len(numeric_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        # Create histograms
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Histogram of {col}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_base64 = plot_to_base64(fig)
        
        # Calculate basic statistics for each feature
        feature_stats = {}
        for col in numeric_cols:
            feature_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
        
        return {
            'filename': filename,
            'numeric_features': numeric_cols.tolist(),
            'plot_base64': plot_base64,
            'feature_statistics': feature_stats,
            'summary': f'Created histograms for {len(numeric_cols)} numeric features'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

# Data Processing Tools

def clean_data(filename: str) -> Dict[str, Any]:
    """Clean dataset by handling missing values and duplicates.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing cleaning results and statistics
    """
    try:
        df = load_dataframe(filename)
        original_shape = df.shape
        
        # Statistics before cleaning
        missing_before = df.isnull().sum().sum()
        duplicates_before = df.duplicated().sum()
        
        # Clean data
        # Remove duplicate rows
        df_cleaned = df.drop_duplicates()
        
        # Handle missing values
        # For numeric columns: fill with median
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        
        # For categorical columns: fill with mode or 'Unknown'
        categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            mode_value = df_cleaned[col].mode()
            fill_value = mode_value.iloc[0] if len(mode_value) > 0 else 'Unknown'
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
        
        # Save cleaned dataset
        cleaned_filename = f"cleaned_{filename}"
        cleaned_path = get_dataset_path(cleaned_filename)
        
        if cleaned_path.suffix.lower() == '.csv':
            df_cleaned.to_csv(cleaned_path, index=False)
        elif cleaned_path.suffix.lower() in ['.xlsx', '.xls']:
            df_cleaned.to_excel(cleaned_path, index=False)
        
        # Statistics after cleaning
        missing_after = df_cleaned.isnull().sum().sum()
        
        return {
            'filename': filename,
            'cleaned_filename': cleaned_filename,
            'original_shape': original_shape,
            'cleaned_shape': df_cleaned.shape,
            'rows_removed': original_shape[0] - df_cleaned.shape[0],
            'duplicates_removed': duplicates_before,
            'missing_values_before': missing_before,
            'missing_values_after': missing_after,
            'missing_values_filled': missing_before - missing_after,
            'cleaning_summary': {
                'numeric_columns_filled': numeric_cols.tolist(),
                'categorical_columns_filled': categorical_cols.tolist()
            }
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def create_feature_histograms(filename: str) -> Dict[str, Any]:
    """Create histograms for all numeric features in a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing histogram plots for all numeric features
    """
    try:
        df = load_dataframe(filename)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric columns found for histogram creation',
                'columns': df.columns.tolist()
            }
        
        # Calculate subplot dimensions
        n_cols = min(3, len(numeric_cols))  # Max 3 columns
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle single subplot case
        if len(numeric_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        # Create histograms
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Histogram of {col}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_base64 = plot_to_base64(fig)
        
        # Calculate basic statistics for each feature
        feature_stats = {}
        for col in numeric_cols:
            feature_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
        
        return {
            'filename': filename,
            'numeric_features': numeric_cols.tolist(),
            'plot_base64': plot_base64,
            'feature_statistics': feature_stats,
            'summary': f'Created histograms for {len(numeric_cols)} numeric features'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def filter_data(filename: str, conditions: str) -> Dict[str, Any]:
    """Filter dataset based on specified conditions.
    
    Args:
        filename: Name of the dataset file
        conditions: Filter conditions in string format (e.g., "age > 25 & income < 50000")
        
    Returns:
        Dict containing filtered data information
    """
    try:
        df = load_dataframe(filename)
        original_shape = df.shape
        
        # Apply filter conditions
        # Note: This is a simple implementation. In production, you'd want more robust parsing
        try:
            # Replace column names with df['column_name'] format for safe evaluation
            safe_conditions = conditions
            for col in df.columns:
                safe_conditions = safe_conditions.replace(col, f"df['{col}']")
            
            # Evaluate the condition
            mask = eval(safe_conditions)
            df_filtered = df[mask]
        except Exception as e:
            return {
                'error': f"Invalid filter condition: {str(e)}",
                'example': "Try conditions like: 'column_name > 10' or 'category == \"value\"'"
            }
        
        # Save filtered dataset
        filtered_filename = f"filtered_{filename}"
        filtered_path = get_dataset_path(filtered_filename)
        
        if filtered_path.suffix.lower() == '.csv':
            df_filtered.to_csv(filtered_path, index=False)
        elif filtered_path.suffix.lower() in ['.xlsx', '.xls']:
            df_filtered.to_excel(filtered_path, index=False)
        
        return {
            'filename': filename,
            'filtered_filename': filtered_filename,
            'conditions': conditions,
            'original_shape': original_shape,
            'filtered_shape': df_filtered.shape,
            'rows_kept': df_filtered.shape[0],
            'rows_removed': original_shape[0] - df_filtered.shape[0],
            'filter_percentage': round((df_filtered.shape[0] / original_shape[0]) * 100, 2),
            'sample_filtered_data': df_filtered.head(5).to_dict('records') if len(df_filtered) > 0 else []
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def create_feature_histograms(filename: str) -> Dict[str, Any]:
    """Create histograms for all numeric features in a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing histogram plots for all numeric features
    """
    try:
        df = load_dataframe(filename)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric columns found for histogram creation',
                'columns': df.columns.tolist()
            }
        
        # Calculate subplot dimensions
        n_cols = min(3, len(numeric_cols))  # Max 3 columns
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle single subplot case
        if len(numeric_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        # Create histograms
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Histogram of {col}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_base64 = plot_to_base64(fig)
        
        # Calculate basic statistics for each feature
        feature_stats = {}
        for col in numeric_cols:
            feature_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
        
        return {
            'filename': filename,
            'numeric_features': numeric_cols.tolist(),
            'plot_base64': plot_base64,
            'feature_statistics': feature_stats,
            'summary': f'Created histograms for {len(numeric_cols)} numeric features'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def sample_data(filename: str, n_rows: int) -> Dict[str, Any]:
    """Create a random sample of the dataset.
    
    Args:
        filename: Name of the dataset file
        n_rows: Number of rows to sample
        
    Returns:
        Dict containing sampling results
    """
    try:
        df = load_dataframe(filename)
        
        if n_rows >= len(df):
            return {
                'error': f"Sample size ({n_rows}) is larger than dataset size ({len(df)})",
                'dataset_size': len(df)
            }
        
        if n_rows <= 0:
            return {
                'error': "Sample size must be positive",
                'dataset_size': len(df)
            }
        
        # Create random sample
        df_sample = df.sample(n=n_rows, random_state=42)
        
        # Save sampled dataset
        sample_filename = f"sample_{n_rows}_{filename}"
        sample_path = get_dataset_path(sample_filename)
        
        if sample_path.suffix.lower() == '.csv':
            df_sample.to_csv(sample_path, index=False)
        elif sample_path.suffix.lower() in ['.xlsx', '.xls']:
            df_sample.to_excel(sample_path, index=False)
        
        return {
            'filename': filename,
            'sample_filename': sample_filename,
            'original_size': len(df),
            'sample_size': n_rows,
            'sample_percentage': round((n_rows / len(df)) * 100, 2),
            'sample_data': df_sample.head(10).to_dict('records'),
            'sample_statistics': {
                'numeric_columns': df_sample.select_dtypes(include=[np.number]).describe().to_dict() if len(df_sample.select_dtypes(include=[np.number]).columns) > 0 else {},
                'categorical_columns': {col: df_sample[col].value_counts().head(5).to_dict() 
                                      for col in df_sample.select_dtypes(include=['object', 'category']).columns}
            }
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def create_feature_histograms(filename: str) -> Dict[str, Any]:
    """Create histograms for all numeric features in a dataset.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Dict containing histogram plots for all numeric features
    """
    try:
        df = load_dataframe(filename)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric columns found for histogram creation',
                'columns': df.columns.tolist()
            }
        
        # Calculate subplot dimensions
        n_cols = min(3, len(numeric_cols))  # Max 3 columns
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle single subplot case
        if len(numeric_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        # Create histograms
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Histogram of {col}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_base64 = plot_to_base64(fig)
        
        # Calculate basic statistics for each feature
        feature_stats = {}
        for col in numeric_cols:
            feature_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
        
        return {
            'filename': filename,
            'numeric_features': numeric_cols.tolist(),
            'plot_base64': plot_base64,
            'feature_statistics': feature_stats,
            'summary': f'Created histograms for {len(numeric_cols)} numeric features'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }