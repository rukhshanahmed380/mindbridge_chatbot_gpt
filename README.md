# MindBridge Chatbot GPT

MindBridge is a safety-aware student support chatbot using retrieval-augmented generation (RAG) with OpenAI chat completions.

## Project Overview

- Loads support resources from a local CSV corpus
- Detects risk level from user input (normal, stress, distress, crisis)
- Retrieves the most relevant chunks from the corpus
- Generates grounded answers using OpenAI
- Provides both:
  - CLI chatbot mode
  - Flask web app mode

## Repository Structure

- `app.py` - Root entry point for the Flask app
- `mindbridge_gpt/app.py` - Flask routes and web integration
- `mindbridge_gpt/chatbot.py` - Core RAG pipeline and CLI chatbot
- `mindbridge_gpt/2_corpus_chunks.csv` - Knowledge corpus
- `mindbridge_gpt/templates/index.html` - Web UI template
- `mindbridge_gpt/static/style.css` - Web UI styles
- `mindbridge_gpt/requirements.txt` - Python dependencies

## Requirements

- Python 3.10+
- OpenAI API key

Install dependencies:

```bash
pip install -r mindbridge_gpt/requirements.txt
```

## Secure API Key Setup

This project does not store API keys in source code.
Set your key using environment variables.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_openai_api_key_here"
```

Command Prompt:

```bat
set OPENAI_API_KEY=your_openai_api_key_here
```

## Run the Project

### Option 1: Run Web App

From repository root:

```bash
python app.py
```

Open in browser:

- http://127.0.0.1:5000

### Option 2: Run CLI Chatbot

```bash
python mindbridge_gpt/chatbot.py
```

CLI commands:

- Type your question to get a response
- `chunks` to inspect loaded chunk groups
- `quit` or `exit` to close

## Notes

- Model can be changed in `mindbridge_gpt/chatbot.py` via `MODEL_ID`
- Answers are intended to be grounded in the provided corpus
- Crisis responses include stronger safety guidance

## Security

- Hardcoded OpenAI keys were removed from code and history
- `.gitignore` now excludes `.venv`, `__pycache__`, and `.pyc` files
- If an old key was exposed previously, revoke and rotate it in OpenAI dashboard
