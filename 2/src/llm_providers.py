# src/llm_providers.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Abstract LLM interface
class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM given a prompt."""
        pass

# Concrete implementation for Anthropic
class AnthropicLLM(LLMProvider):
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", **kwargs):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(**kwargs)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install the Anthropic Python package: pip install anthropic")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

# Concrete implementation for OpenAI
class OpenAILLM(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        try:
            from openai import OpenAI
            self.client = OpenAI(**kwargs)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install the OpenAI Python package: pip install openai")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Unified prompt template class
class PromptTemplate:
    @classmethod
    def from_file(cls, file_path: str, input_variables: list):
        with open(file_path, 'r') as file:
            template = file.read()
        return cls(template, input_variables)
    
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        """Format the template with the provided variables."""
        formatted_prompt = self.template
        for var in self.input_variables:
            if var in kwargs:
                formatted_prompt = formatted_prompt.replace(f"{{{var}}}", str(kwargs[var]))
        return formatted_prompt

# The refactored QueryRouter class
class QueryRouter:
    def __init__(self, llm_provider: LLMProvider, prompt_path: str):
        self.llm = llm_provider
        self.prompt_template = PromptTemplate.from_file(
            prompt_path,
            input_variables=["KNOWLEDGE_BASE", "AVAILABLE_FUNCTIONS", "USER_QUERY"]
        )
    
    def classify_query(self, user_query: str, knowledge_base: str, available_functions: str) -> Dict[str, Any]:
        """
        Classify the query type and determine how to process it.
        
        Args:
            user_query: The user's query to be classified
            knowledge_base: Description of the knowledge base available
            available_functions: List of available tools/functions
            
        Returns:
            A dictionary containing the classification results
        """
        formatted_prompt = self.prompt_template.format(
            KNOWLEDGE_BASE=knowledge_base,
            AVAILABLE_FUNCTIONS=available_functions,
            USER_QUERY=user_query
        )
        
        response = self.llm.generate_response(formatted_prompt)
        
        # Parse the response to extract classification
        # This is a simplified example - you'd need proper parsing logic
        classification = self._parse_classification(response)
        return classification
    
    def _parse_classification(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract the classification details."""
        # Implement parsing logic based on your expected response format
        # This is just a placeholder
        if "direct knowledge" in response.lower():
            return {"type": "direct_knowledge", "reasoning": response}
        elif "research" in response.lower():
            return {"type": "research_needed", "reasoning": response}
        elif "tool" in response.lower():
            return {"type": "tool_required", "reasoning": response}
        else:
            return {"type": "unknown", "reasoning": response}