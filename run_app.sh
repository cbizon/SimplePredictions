#!/bin/bash
# Script to run the SimplePredictions evaluation viewer Flask app

echo "Starting SimplePredictions Evaluation Viewer..."
echo "Navigate to http://localhost:5000 in your browser"
echo "Press Ctrl+C to stop the server"
echo ""

uv run simplepredictions-web
