research_assistant/
├── data/
│   └── documents/            # Your document corpus
├── prompts/                  # Store prompt templates
├── src/
│   ├── __init__.py
│   ├── agent.py              # Main agent implementation
│   ├── llm_providers.py      # LLM abstraction layer (already started)
│   ├── rag_pipeline.py       # Your existing RAG implementation
│   ├── router.py             # Query router (already started)
│   ├── tools/                # External tools implementation
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   └── web_search.py
│   └── dspy_modules/         # DSPy components
│       ├── __init__.py
│       ├── signatures.py
│       └── evaluators.py
├── main.py                   # Entry point
├── requirements.txt
└── README.md