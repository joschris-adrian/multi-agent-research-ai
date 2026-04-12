#!/bin/bash
# Run this once after docker-compose up to pull the Ollama model

echo "Waiting for Ollama to be ready..."
sleep 5

docker exec ollama ollama pull llama3.2

echo "Model ready! Open http://localhost:8501"
