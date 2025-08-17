#!/usr/bin/env python3
"""
MCP Client for Data Analysis Chatbot

Handles communication with the MCP data analysis server and integrates
with the Anthropic Claude API for natural language processing.
"""

import json
import asyncio
import subprocess
import os
import sys
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MCPDataClient:
    """Client for communicating with the MCP data analysis server."""
    
    def __init__(self, server_path: Optional[str] = None):
        self.server_path = server_path or str(Path(__file__).parent.parent / "server" / "main.py")
        self.anthropic_client = None
        self.available_tools = {}
        
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.anthropic_client = Anthropic(api_key=api_key)
        
        # Initialize tool definitions
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize the available MCP tools for Claude."""
        self.available_tools = {
            "list_available_datasets": {
                "name": "list_available_datasets",
                "description": "List all available datasets in the datasets directory",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            "load_dataset": {
                "name": "load_dataset", 
                "description": "Load a dataset and return basic information about it",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the dataset file to load"
                        }
                    },
                    "required": ["filename"]
                }
            },
            "dataset_info": {
                "name": "dataset_info",
                "description": "Get comprehensive information about a dataset including statistics and sample data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the dataset file"
                        }
                    },
                    "required": ["filename"]
                }
            },
            "describe_data": {
                "name": "describe_data",
                "description": "Generate descriptive statistics for a dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the dataset file"
                        }
                    },
                    "required": ["filename"]
                }
            },
            "correlation_analysis": {
                "name": "correlation_analysis",
                "description": "Perform correlation analysis on numeric columns and generate a correlation heatmap",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the dataset file"
                        }
                    },
                    "required": ["filename"]
                }
            },
            "group_analysis": {
                "name": "group_analysis",
                "description": "Perform group-by analysis on a dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the dataset file"
                        },
                        "group_by_column": {
                            "type": "string",
                            "description": "Column name to group by"
                        }
                    },
                    "required": ["filename", "group_by_column"]
                }
            },
            "simple_plot": {
                "name": "simple_plot",
                "description": "Generate simple plots for data visualization",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the dataset file"
                        },
                        "plot_type": {
                            "type": "string",
                            "description": "Type of plot",
                            "enum": ["histogram", "scatter", "bar", "line", "box"]
                        },
                        "x_column": {
                            "type": "string",
                            "description": "Column for x-axis"
                        },
                        "y_column": {
                            "type": "string",
                            "description": "Column for y-axis (optional for some plot types)"
                        }
                    },
                    "required": ["filename", "plot_type", "x_column"]
                }
            },
            "clean_data": {
                "name": "clean_data",
                "description": "Clean dataset by handling missing values and duplicates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the dataset file"
                        }
                    },
                    "required": ["filename"]
                }
            },
            "filter_data": {
                "name": "filter_data",
                "description": "Filter dataset based on specified conditions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string", 
                            "description": "Name of the dataset file"
                        },
                        "conditions": {
                            "type": "string",
                            "description": "Filter conditions (e.g., 'age > 25 & income < 50000')"
                        }
                    },
                    "required": ["filename", "conditions"]
                }
            },
            "sample_data": {
                "name": "sample_data",
                "description": "Create a random sample of the dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the dataset file"
                        },
                        "n_rows": {
                            "type": "integer",
                            "description": "Number of rows to sample"
                        }
                    },
                    "required": ["filename", "n_rows"]
                }
            },
            "create_feature_histograms": {
                "name": "create_feature_histograms",
                "description": "Create histograms for all numeric features in a dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the dataset file"
                        }
                    },
                    "required": ["filename"]
                }
            }
        }
    
    def call_mcp_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call an MCP tool by executing the server script."""
        try:
            # For this prototype, we'll directly import and call the server functions
            # In a real implementation, this would use proper MCP protocol
            
            import sys
            # Add the project root to sys.path so we can import server.main
            project_root = str(Path(self.server_path).parent.parent)
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            # Import the callable tools
            from server.tools import (
                list_available_datasets, load_dataset, dataset_info, describe_data,
                correlation_analysis, group_analysis, simple_plot, clean_data,
                filter_data, sample_data, create_feature_histograms
            )
            
            # Map tool names to functions
            tool_functions = {
                "list_available_datasets": list_available_datasets,
                "load_dataset": load_dataset,
                "dataset_info": dataset_info,
                "describe_data": describe_data,
                "correlation_analysis": correlation_analysis,
                "group_analysis": group_analysis,
                "simple_plot": simple_plot,
                "clean_data": clean_data,
                "filter_data": filter_data,
                "sample_data": sample_data,
                "create_feature_histograms": create_feature_histograms
            }
            
            if tool_name not in tool_functions:
                return {"error": f"Unknown tool: {tool_name}"}
            
            # Call the function with provided arguments
            result = tool_functions[tool_name](**kwargs)
            return result
            
        except Exception as e:
            import traceback
            return {
                "error": f"Error calling MCP tool '{tool_name}': {str(e)}",
                "tool_name": tool_name,
                "arguments": kwargs,
                "traceback": traceback.format_exc()
            }
    
    def _truncate_conversation(self, messages: List[Dict[str, str]], max_messages: int = 20) -> List[Dict[str, str]]:
        """Truncate conversation history to manage token limits."""
        if len(messages) <= max_messages:
            return messages
        
        # Keep the first message (often contains important context) and the most recent messages
        return [messages[0]] + messages[-(max_messages-1):]
    
    def chat_with_claude(self, messages: List[Dict[str, str]], current_dataset: Optional[str] = None) -> Dict[str, Any]:
        """Send messages to Claude with MCP tools available."""
        if not self.anthropic_client:
            return {
                "error": "Anthropic API client not initialized. Check your API key.",
                "response": "Please configure your Anthropic API key in the .env file."
            }
        
        try:
            # Prepare system message with tool information
            system_message = f"""You are a helpful data analysis assistant with access to powerful MCP tools for analyzing datasets. 

Current dataset: {current_dataset or 'None selected'}

You have access to the following tools for data analysis:
- list_available_datasets(): List all available datasets
- load_dataset(filename): Load a dataset and get basic info
- dataset_info(filename): Get detailed dataset information  
- describe_data(filename): Get descriptive statistics
- correlation_analysis(filename): Perform correlation analysis with heatmap
- group_analysis(filename, group_by_column): Perform group-by analysis
- simple_plot(filename, plot_type, x_column, y_column): Generate plots
- clean_data(filename): Clean dataset (handle missing values, duplicates)
- filter_data(filename, conditions): Filter dataset based on conditions
- sample_data(filename, n_rows): Create random sample of dataset

When users ask questions about data analysis:
1. Determine which tools to call based on their request
2. Call the appropriate tools with the right parameters
3. Interpret the results and provide helpful insights
4. If plots are generated, mention that visualizations are available

Always use the exact dataset filename when calling tools. Be helpful and provide detailed explanations of the analysis results."""

            # Convert tools to Claude format
            tools = list(self.available_tools.values())
            
            # Truncate conversation to manage token limits
            truncated_messages = self._truncate_conversation(messages)
            
            # Make the API call
            response = self.anthropic_client.messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
                max_tokens=4000,
                system=system_message,
                tools=tools,
                messages=truncated_messages
            )
            
            # Process the response
            result = {
                "response": "",
                "tool_calls": [],
                "plots": []
            }
            
            # Handle the response content
            tool_results = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    result["response"] += content_block.text
                elif content_block.type == "tool_use":
                    # Execute the tool call
                    tool_result = self.call_mcp_tool(
                        content_block.name,
                        **content_block.input
                    )
                    
                    result["tool_calls"].append({
                        "tool": content_block.name,
                        "input": content_block.input,
                        "result": tool_result
                    })
                    
                    # Extract plots if available
                    if "plot_base64" in tool_result:
                        result["plots"].append(tool_result["plot_base64"])
                    
                    # Prepare tool result for Claude
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": json.dumps(tool_result, default=str)
                    })
            
            # If there were tool calls, get the final response from Claude
            if tool_results and response.stop_reason == "tool_use":
                tool_messages = truncated_messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": tool_results}
                ]
                
                final_response = self.anthropic_client.messages.create(
                    model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
                    max_tokens=4000,
                    system=system_message,
                    tools=tools,
                    messages=tool_messages
                )
                
                # Process final response
                for final_content in final_response.content:
                    if final_content.type == "text":
                        result["response"] += final_content.text
                    elif final_content.type == "tool_use":
                        # Handle additional tool calls if any
                        additional_tool_result = self.call_mcp_tool(
                            final_content.name,
                            **final_content.input
                        )
                        
                        result["tool_calls"].append({
                            "tool": final_content.name,
                            "input": final_content.input,
                            "result": additional_tool_result
                        })
                        
                        if "plot_base64" in additional_tool_result:
                            result["plots"].append(additional_tool_result["plot_base64"])
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "error": f"Error communicating with Claude: {str(e)}",
                "response": f"Sorry, I encountered an error while processing your request. Error: {str(e)}",
                "tool_calls": [],
                "plots": [],
                "traceback": traceback.format_exc()
            }

# Singleton instance for use in Streamlit
mcp_client = MCPDataClient()