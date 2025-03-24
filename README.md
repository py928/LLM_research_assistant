# LLM-Powered Research Assistant

A RAG-based AI research assistant that intelligently answers user queries by retrieving relevant documents and generating responses.

## Features

- **Autonomous Decision Making**: Assistant intelligently decides whether to:
  - Answer directly from memory
  - Retrieve additional context before answering
  - Call specialized tools when needed

- **Advanced RAG Pipeline**: Retrieval-augmented generation with vector similarity search

- **Multi-Step Reasoning**: Structured thinking process for complex queries

- **Tool Integration**: Calculator and web search functionality

- **DSPy Integration**: For prompt engineering and evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com//ako1983/research-assistant.git
cd research-assistant

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
