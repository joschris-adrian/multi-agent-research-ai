import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Force Ollama for all tests so .env values don't bleed in.
# Must be set before any src module is imported.
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["GEMINI_API_KEY"] = ""