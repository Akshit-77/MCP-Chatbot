#!/usr/bin/env python3
"""
MCP Data Analysis Chatbot - Streamlit Frontend

A Streamlit-based chat interface that connects to the MCP data analysis server
and provides natural language interaction with dataset analysis capabilities.
"""

import streamlit as st
import json
import uuid
import base64
import os
import io
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

# Import MCP client
from mcp_client import mcp_client

# Load environment variables
load_dotenv()

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Configuration
SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
SESSIONS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

# Anthropic API configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

# File upload configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
ALLOWED_FILE_TYPES = os.getenv("ALLOWED_FILE_TYPES", "csv,xlsx,xls").split(",")

# Initialize Anthropic client
if ANTHROPIC_API_KEY:
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    anthropic_client = None

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None

def save_session():
    """Save current session to JSON file."""
    if not st.session_state.messages:
        return
    
    session_data = {
        'session_id': st.session_state.session_id,
        'created_at': datetime.now().isoformat(),
        'current_dataset': st.session_state.current_dataset,
        'messages': st.session_state.messages
    }
    
    session_file = SESSIONS_DIR / f"{st.session_state.session_id}.json"
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=2, cls=CustomJSONEncoder)

def load_session(session_id: str) -> bool:
    """Load session from JSON file."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    
    if not session_file.exists():
        return False
    
    try:
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        st.session_state.session_id = session_data['session_id']
        st.session_state.messages = session_data.get('messages', [])
        st.session_state.current_dataset = session_data.get('current_dataset')
        return True
    except Exception as e:
        st.error(f"Error loading session: {str(e)}")
        return False

def get_available_sessions() -> List[Dict[str, Any]]:
    """Get list of available sessions."""
    sessions = []
    
    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            sessions.append({
                'id': session_data['session_id'],
                'created_at': session_data.get('created_at', ''),
                'dataset': session_data.get('current_dataset', 'No dataset'),
                'message_count': len(session_data.get('messages', []))
            })
        except Exception:
            continue
    
    # Sort by creation time (newest first)
    sessions.sort(key=lambda x: x['created_at'], reverse=True)
    return sessions

def upload_dataset(uploaded_file) -> bool:
    """Handle dataset file upload."""
    if uploaded_file is None:
        return False
    
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
        return False
    
    # Check file type
    file_extension = Path(uploaded_file.name).suffix.lower()[1:]  # Remove the dot
    if file_extension not in ALLOWED_FILE_TYPES:
        st.error(f"Unsupported file type. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}")
        return False
    
    try:
        # Save the uploaded file
        file_path = DATASETS_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.current_dataset = uploaded_file.name
        st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")
        return True
        
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return False

def get_available_datasets() -> List[str]:
    """Get list of available datasets."""
    datasets = []
    for file_path in DATASETS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower()[1:] in ALLOWED_FILE_TYPES:
            datasets.append(file_path.name)
    return sorted(datasets)

def send_message_to_claude(message: str) -> Dict[str, Any]:
    """Send message to Claude with MCP tools context."""
    try:
        # Build conversation history
        conversation = []
        for msg in st.session_state.messages:
            conversation.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add the new user message
        conversation.append({
            "role": "user",
            "content": message
        })
        
        # Use MCP client to handle the conversation
        result = mcp_client.chat_with_claude(
            messages=conversation,
            current_dataset=st.session_state.current_dataset
        )
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "response": f"Error communicating with Claude: {str(e)}",
            "tool_calls": [],
            "plots": []
        }

def render_chat_interface():
    """Render the main chat interface."""
    st.header("💬 Chat with your Data")
    
    # Display current dataset info
    if st.session_state.current_dataset:
        st.info(f"📊 Current dataset: **{st.session_state.current_dataset}**")
    else:
        st.warning("📂 No dataset selected. Upload a file or select from available datasets.")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display plots if available
                plots = message.get("plots", [])
                if plots:
                    for plot_base64 in plots:
                        if plot_base64:
                            try:
                                plot_data = base64.b64decode(plot_base64)
                                st.image(plot_data, caption="Generated Visualization")
                            except Exception as e:
                                st.error(f"Error displaying plot: {str(e)}")
                
                # Display legacy plot format for backward compatibility
                if "plot_base64" in message and message["plot_base64"]:
                    try:
                        plot_data = base64.b64decode(message["plot_base64"])
                        st.image(plot_data, caption="Generated Plot")
                    except Exception as e:
                        st.error(f"Error displaying plot: {str(e)}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from Claude
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                result = send_message_to_claude(prompt)
            
            # Handle error case
            if "error" in result and not result.get("response"):
                st.error(f"Error: {result['error']}")
                # Show additional debugging info
                with st.expander("Debug Information"):
                    st.json(result)
                return
            
            # Display the response
            response_text = result.get("response", "")
            if response_text:
                st.write(response_text)
            
            # Display any plots
            plots = result.get("plots", [])
            for plot_base64 in plots:
                if plot_base64:
                    try:
                        plot_data = base64.b64decode(plot_base64)
                        st.image(plot_data, caption="Generated Visualization")
                    except Exception as e:
                        st.error(f"Error displaying plot: {str(e)}")
            
            # Display tool call information (for debugging)
            tool_calls = result.get("tool_calls", [])
            if tool_calls:
                if st.checkbox("Show tool details", key=f"tools_{len(st.session_state.messages)}"):
                    with st.expander("Tool Calls"):
                        for tool_call in tool_calls:
                            st.write(f"**Tool:** {tool_call['tool']}")
                            st.write(f"**Input:** {tool_call['input']}")
                            if "error" not in tool_call["result"]:
                                st.success("Tool executed successfully")
                                # Show abbreviated result
                                if isinstance(tool_call["result"], dict):
                                    if "shape" in tool_call["result"]:
                                        st.write(f"Dataset shape: {tool_call['result']['shape']}")
                                    if "filename" in tool_call["result"]:
                                        st.write(f"Filename: {tool_call['result']['filename']}")
                            else:
                                st.error(f"Tool error: {tool_call['result']['error']}")
                                if "traceback" in tool_call["result"]:
                                    with st.expander("Error Details"):
                                        st.text(tool_call["result"]["traceback"])
            
            # Always show debug info if there are any issues
            if "error" in result or not response_text:
                with st.expander("Debug Information"):
                    st.json(result)
            
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "plots": plots,
                "tool_calls": tool_calls
            })
        
        # Save session after each interaction
        save_session()
        
        # Rerun to update the chat display
        st.rerun()

def render_sidebar():
    """Render the sidebar with session management and dataset controls."""
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Session Management
        st.subheader("📝 Session Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Session", type="primary"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.current_dataset = None
                st.rerun()
        
        with col2:
            if st.button("Save Session"):
                save_session()
                st.success("Session saved!")
        
        # Load existing sessions
        sessions = get_available_sessions()
        if sessions:
            st.subheader("📂 Load Session")
            session_options = [f"{s['id'][:8]}... ({s['dataset']}, {s['message_count']} msgs)" 
                             for s in sessions]
            
            selected_session = st.selectbox(
                "Choose session",
                options=[None] + session_options,
                format_func=lambda x: "Select a session..." if x is None else x
            )
            
            if selected_session and st.button("Load Selected"):
                session_id = sessions[session_options.index(selected_session)]['id']
                if load_session(session_id):
                    st.success("Session loaded!")
                    st.rerun()
        
        st.divider()
        
        # Dataset Management
        st.subheader("📊 Dataset Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=ALLOWED_FILE_TYPES,
            help=f"Supported formats: {', '.join(ALLOWED_FILE_TYPES)}. Max size: {MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file and st.button("Upload File"):
            upload_dataset(uploaded_file)
            st.rerun()
        
        # Available datasets
        available_datasets = get_available_datasets()
        if available_datasets:
            st.subheader("📁 Available Datasets")
            selected_dataset = st.selectbox(
                "Select dataset",
                options=[None] + available_datasets,
                format_func=lambda x: "Choose a dataset..." if x is None else x
            )
            
            if selected_dataset and st.button("Use Dataset"):
                st.session_state.current_dataset = selected_dataset
                st.success(f"Dataset '{selected_dataset}' selected!")
                st.rerun()
        
        st.divider()
        
        # App Info
        st.subheader("ℹ️ App Info")
        st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        st.write(f"**Available Datasets:** {len(available_datasets)}")

def main():
    """Main application function."""
    st.set_page_config(
        page_title="MCP Data Analysis Chatbot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 MCP Data Analysis Chatbot")
    st.caption("Analyze your datasets with natural language powered by Claude and MCP")
    
    # Initialize session state
    init_session_state()
    
    # Check if Anthropic API key is configured
    if not ANTHROPIC_API_KEY:
        st.error("⚠️ Anthropic API key not found. Please configure ANTHROPIC_API_KEY in your .env file.")
        st.info("Copy .env.template to .env and add your API key.")
        return
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat interface
    render_chat_interface()
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit, Claude API, and MCP • Data stays local on your machine")

if __name__ == "__main__":
    main()