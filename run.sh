#!/bin/bash
# Activate virtual environment
source venv/bin/activate

# Run the server
echo "Starting Tulane University Chatbot on http://localhost:8000"
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
