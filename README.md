
# LLM-Powered Research Assistant

An AI-powered research assistant that leverages Retrieval-Augmented Generation (RAG) to provide accurate responses to user queries by retrieving relevant documents and reasoning through complex questions.

## Features

- **Smart Query Routing**: Autonomously decides whether to answer directly from knowledge, retrieve additional context, or use specialized tools
- **RAG Pipeline**: Retrieves relevant documents to enhance responses with accurate, up-to-date information
- **Multi-step Reasoning**: Uses DSPy for structured reasoning to break down complex queries
- **Tool Integration**: Can utilize calculators, web search, and other external tools when needed
- **Evaluation Framework**: Measures response quality and relevance using DSPy's evaluation capabilities

## Architecture

```ascii
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User Interface │────▶│  Query Router   │────▶│  RAG Pipeline   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │  ▲                     │
                               ▼  │                     ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   Tools (Calc,  │     │  Vector Store   │
                        │   Web Search)   │     │   (ChromaDB)    │
                        └─────────────────┘     └─────────────────┘
                               │  ▲                     │
                               ▼  │                     ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   LLM Provider  │◀───▶│  DSPy Modules   │
                        │ (OpenAI/Claude) │     │    & Metrics    │
                        └─────────────────┘     └─────────────────┘
```

## Setup & Installation

1. Clone the repository

```bash
git clone https://github.com/[your-username]/llm-research-assistant.git
cd llm-research-assistant
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up environment variables

```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"  # If using Claude
```

4. Prepare your data

```bash
python src/vectorstore_builder.py
```

## Usage

```python
from src.agent import ResearchAssistant
from src.llm_providers import OpenAILLM
from src.rag_pipeline import RAGPipeline

# Initialize components
llm = OpenAILLM(model_name="gpt-3.5-turbo")
rag = RAGPipeline()
rag.initialize()
retriever = rag.get_retriever()

# Create and use the assistant
assistant = ResearchAssistant(llm_provider=llm, retriever=retriever)
response = assistant.process_query("How do I fix Wi-Fi connection issues?")
print(response["response"])
```

## Project Structure

```
llm-research-assistant/
├── data/
│   ├── raw/                # Original dataset files
│   ├── processed/          # Cleaned CSV files
│   └── vector_stores/      # ChromaDB vector stores
├── prompts/
│   └── query_classification_prompt_template.txt  # LLM prompts
├── src/
│   ├── agent.py            # Main assistant logic
│   ├── llm_providers.py    # LLM abstraction layer
│   ├── rag_pipeline.py     # Document retrieval system
│   ├── router.py           # Query routing logic
│   ├── tools/              # External tool integrations
│   └── dspy_modules/       # DSPy components
├── tests/                  # Test cases
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## Requirements

- Python 3.8+
- LangChain
- DSPy
- ChromaDB
- OpenAI or Anthropic API access

## Evaluation

The system uses DSPy's evaluation framework to assess:

- Answer correctness
- Context relevance
- Reasoning quality

## License

MIT

## Acknowledgements

- Data sourced from [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- Built for an assessment

```
