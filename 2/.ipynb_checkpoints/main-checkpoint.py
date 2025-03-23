# In main.py
import os
from src.llm_providers import AnthropicLLM, OpenAILLM
from src.agent import ResearchAssistant
from src.tools.calculator import Calculator
from src.tools.web_search import WebSearch
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    # Setup environment variables
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"
    
    # Initialize LLM provider (switch as needed)
    llm = OpenAILLM()  # or AnthropicLLM()
    
    # Initialize document processing pipeline
    loader = DirectoryLoader("data/documents/", glob="**/*.pdf")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Initialize vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Initialize tools
    tools = {
        "calculator": Calculator(),
        "web_search": WebSearch()
    }
    
    # Initialize agent
    agent = ResearchAssistant(
        llm_provider=llm,
        retriever=retriever,
        tools=tools
    )
    
    # Interactive loop
    print("Research Assistant initialized. Type 'exit' to quit.")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break
            
        result = agent.process_query(query)
        print("\nAnswer:", result["response"])
        print("\nQuery type:", result["query_type"])

if __name__ == "__main__":
    main()