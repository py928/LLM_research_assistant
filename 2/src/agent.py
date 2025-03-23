# !src/agent.py
from typing import Dict, List, Any, Optional
from .llm_providers import LLMProvider
from .router import QueryRouter
import dspy

class ResearchAssistant:
    def __init__(
        self, 
        llm_provider: LLMProvider,
        retriever,  # Your ChromaDB retriever
        tools: Dict[str, Any] = None,
        prompt_dir: str = "prompts/"
    ):
        self.llm = llm_provider
        self.retriever = retriever
        self.tools = tools or {}
        self.router = QueryRouter(
            llm_provider=llm_provider,
            prompt_path=f"{prompt_dir}/router_prompt.txt"
            #query_classification_prompt_template.txt
        )
        self.dspy_llm = dspy.OpenAI(model="gpt-4o")  #  Anthropic model
        dspy.settings.configure(lm=self.dspy_llm)
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for processing user queries.
        
        Args:
            query: The user's question or request
            
        Returns:
            A dictionary containing the response and metadata
        """
        # 1. Classify the query
        classification = self.router.classify_query(
            user_query=query,
            knowledge_base="Research papers and technical documents",
            available_functions=str(list(self.tools.keys()))
        )
        
        # 2. Route to appropriate handler
        query_type = classification.get("type", "unknown")
        
        if query_type == "direct_knowledge":
            response = self.handle_direct_knowledge(query)
        elif query_type == "research_needed":
            response = self.handle_research(query)
        elif query_type == "tool_required":
            response = self.handle_tool_call(query, classification)
        else:
            # Default to research if classification fails
            response = self.handle_research(query)
            
        # 3. Return response with metadata
        return {
            "query": query,
            "response": response,
            "query_type": query_type,
            "classification": classification
        }
        
    def handle_direct_knowledge(self, query: str) -> str:
        """Handle queries that can be answered directly from LLM knowledge."""
        prompt = f"""Answer the following question using your existing knowledge:
        
        Question: {query}
        
        Provide a comprehensive, accurate response based on what you know.
        """
        return self.llm.generate_response(prompt)
        
    def handle_research(self, query: str) -> str:
        """Handle queries that require retrieving documents."""
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Format documents for context
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                              for i, doc in enumerate(docs)])
        
        # Create augmented prompt
        prompt = f"""Answer the following question using the provided context and your knowledge:
        
        Question: {query}
        
        Context:
        {context}
        
        Provide a comprehensive, accurate response based on the context and what you know.
        If the context doesn't contain relevant information, state that clearly.
        """
        return self.llm.generate_response(prompt)
        
    def handle_tool_call(self, query: str, classification: Dict[str, Any]) -> str:
        """Handle queries that require calling external tools."""
        # Identify which tool to use (could be enhanced with LLM)
        tool_name = None
        for tool in self.tools.keys():
            if tool.lower() in query.lower():
                tool_name = tool
                break
                
        if not tool_name:
            # Default to the first tool or return research results
            if self.tools:
                tool_name = list(self.tools.keys())[0]
            else:
                return self.handle_research(query)
                
        # Execute the tool
        tool = self.tools[tool_name]
        try:
            tool_result = tool.execute(query)
            
            # Create response with tool result
            prompt = f"""The user asked: {query}
            
            I used the {tool_name} tool and got this result:
            {tool_result}
            
            Based on this result, provide a helpful response to the user's query.
            """
            return self.llm.generate_response(prompt)
        except Exception as e:
            # Fallback to research if tool execution fails
            return f"I tried to use {tool_name} but encountered an error. Let me research this instead.\n\n" + self.handle_research(query)