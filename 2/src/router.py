# !router.py
#  QueryRouter class
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
        
        # ############# Parse the response to extract classification
        # This is a simplified example - you'd need proper parsing logic
        classification = self._parse_classification(response)
        return classification
    
    def _parse_classification(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract the classification details."""
        # Implement parsing logic based on your expected response format
        ############## This is just a placeholder
        if "direct knowledge" in response.lower():
            return {"type": "direct_knowledge", "reasoning": response}
        elif "research" in response.lower():
            return {"type": "research_needed", "reasoning": response}
        elif "tool" in response.lower():
            return {"type": "tool_required", "reasoning": response}
        else:
            return {"type": "unknown", "reasoning": response}