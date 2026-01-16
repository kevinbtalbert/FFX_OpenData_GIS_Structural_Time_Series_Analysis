# ###########################################################################
#
#  Azure OpenAI Chatbot Integration
#  AI Assistant for Real Estate Forecasting Insights
#
# ###########################################################################

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

try:
    from openai import AzureOpenAI
except ImportError:
    print("Warning: openai package not installed. Install with: pip install openai")
    AzureOpenAI = None


class RealEstateChatbot:
    """
    AI Assistant for answering executive questions about real estate forecasts.
    Uses Azure OpenAI for natural language understanding and generation.
    """
    
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None
    ):
        """
        Initialize the chatbot with Azure OpenAI credentials.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT env var)
            api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)
            api_version: API version to use (or set AZURE_OPENAI_API_VERSION env var)
            deployment_name: Name of the deployed model (or set AZURE_OPENAI_DEPLOYMENT env var)
        """
        # Get credentials from environment if not provided
        self.azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        self.deployment_name = deployment_name or os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')
        
        # Validate credentials
        if not self.azure_endpoint or not self.api_key:
            raise ValueError(
                "Azure OpenAI credentials not provided. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables, "
                "or pass them as arguments."
            )
        
        # Initialize Azure OpenAI client
        if AzureOpenAI is None:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # Conversation history
        self.conversation_history = []
        
        # System prompt
        self.system_prompt = """You are an AI assistant specializing in real estate analytics and forecasting for Fairfax County, Virginia. 
You help executives understand property value trends, forecasts, and revenue risks.

Your capabilities include:
- Analyzing historical property value trends
- Explaining forecast predictions and confidence intervals
- Identifying districts with highest/lowest growth
- Assessing revenue risk and opportunities
- Providing strategic insights based on data

When answering questions:
- Be concise and executive-focused
- Use specific numbers and percentages when available
- Highlight key insights and actionable recommendations
- Explain uncertainty and confidence levels clearly
- Format large numbers with commas (e.g., $1,234,567)

You have access to real estate forecast data and can answer questions about:
- Property values by district
- Future predictions (6-12 months ahead)
- Historical trends
- Revenue projections
- Risk assessments
"""
    
    def chat(
        self,
        user_message: str,
        forecast_data: Optional[Dict] = None,
        context: Optional[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 800
    ) -> str:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            user_message: User's question or message
            forecast_data: Current forecast data to provide as context
            context: Additional context (e.g., selected district, date range)
            temperature: Sampling temperature (0-1, higher = more creative)
            max_tokens: Maximum tokens in response
        
        Returns:
            Chatbot's response
        """
        # Build context message
        context_message = self._build_context_message(forecast_data, context)
        
        # Build messages for API call
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add context if available
        if context_message:
            messages.append({"role": "system", "content": context_message})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        # Call Azure OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            assistant_message = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Keep only last 10 messages to avoid token limits
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return assistant_message
            
        except Exception as e:
            return f"Error communicating with Azure OpenAI: {str(e)}"
    
    def _build_context_message(
        self,
        forecast_data: Optional[Dict],
        context: Optional[Dict]
    ) -> str:
        """Build a context message with current forecast data."""
        context_parts = []
        
        if context:
            context_parts.append("Current Context:")
            if 'district' in context:
                context_parts.append(f"- District: {context['district']}")
            if 'date_range' in context:
                context_parts.append(f"- Date Range: {context['date_range']}")
        
        if forecast_data:
            context_parts.append("\nCurrent Forecast Data:")
            
            if 'district' in forecast_data:
                context_parts.append(f"- District: {forecast_data['district']}")
            
            if 'predictions' in forecast_data and len(forecast_data['predictions']) > 0:
                context_parts.append(f"- Forecast Period: {len(forecast_data['predictions'])} periods ahead")
                
                # Add summary statistics
                if 'total_predicted_value' in forecast_data:
                    total = forecast_data['total_predicted_value']
                    context_parts.append(f"- Total Predicted Value: ${total:,.0f}")
                
                if 'mean_predicted_value' in forecast_data:
                    mean = forecast_data['mean_predicted_value']
                    context_parts.append(f"- Mean Predicted Value: ${mean:,.0f}")
                
                # Add first and last predictions
                first_pred = forecast_data['predictions'][0]
                last_pred = forecast_data['predictions'][-1]
                
                context_parts.append(f"\nFirst Prediction ({first_pred['date']}):")
                context_parts.append(f"  - Value: ${first_pred['predicted_value']:,.0f}")
                context_parts.append(f"  - Range: ${first_pred['lower_bound']:,.0f} - ${first_pred['upper_bound']:,.0f}")
                
                context_parts.append(f"\nLast Prediction ({last_pred['date']}):")
                context_parts.append(f"  - Value: ${last_pred['predicted_value']:,.0f}")
                context_parts.append(f"  - Range: ${last_pred['lower_bound']:,.0f} - ${last_pred['upper_bound']:,.0f}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_suggested_questions(self, forecast_data: Optional[Dict] = None) -> List[str]:
        """
        Get suggested questions based on current context.
        
        Returns:
            List of suggested questions
        """
        questions = [
            "What is the predicted property value for the next 6 months?",
            "Which districts show the highest growth potential?",
            "What is the revenue risk if values decline?",
            "How confident are these predictions?",
            "What factors could impact these forecasts?",
            "Compare this district to the county average",
            "What trends do you see in the historical data?",
            "Should we be concerned about any districts?"
        ]
        
        if forecast_data and 'district' in forecast_data:
            district = forecast_data['district']
            questions.insert(0, f"What's the outlook for district {district}?")
        
        return questions


class MockChatbot(RealEstateChatbot):
    """
    Mock chatbot for testing without Azure OpenAI credentials.
    Returns canned responses based on keywords.
    """
    
    def __init__(self):
        """Initialize mock chatbot without credentials."""
        self.conversation_history = []
        self.system_prompt = "Mock chatbot for testing"
    
    def chat(
        self,
        user_message: str,
        forecast_data: Optional[Dict] = None,
        context: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """Return mock response based on keywords."""
        message_lower = user_message.lower()
        
        # Keyword-based responses
        if 'predict' in message_lower or 'forecast' in message_lower:
            if forecast_data and 'total_predicted_value' in forecast_data:
                total = forecast_data['total_predicted_value']
                return f"Based on the forecast model, the predicted total property value is ${total:,.0f}. This represents the aggregate assessed value across all properties in the selected area over the forecast period."
            return "The forecast shows steady growth in property values over the next 6 months, with a confidence interval of 95%."
        
        elif 'risk' in message_lower:
            return "The main revenue risks include: (1) Economic downturn affecting property values, (2) Changes in tax assessment policies, (3) Market volatility in specific districts. I recommend monitoring high-value districts closely."
        
        elif 'district' in message_lower or 'area' in message_lower:
            return "Different districts show varying growth patterns. Urban districts near commercial centers tend to show higher appreciation rates, while rural districts show more stable but slower growth."
        
        elif 'confidence' in message_lower or 'certain' in message_lower:
            return "The forecast model provides 95% confidence intervals. This means we can be 95% confident that actual values will fall within the predicted range. The model accounts for historical trends and seasonal patterns."
        
        elif 'trend' in message_lower or 'historical' in message_lower:
            return "Historical data shows consistent year-over-year growth in property values, with some seasonal variation. Recent trends indicate strong demand in residential areas."
        
        else:
            return "I can help you understand property value forecasts, revenue risks, and district trends. What specific aspect would you like to explore?"


def create_chatbot(use_mock: bool = False, **kwargs) -> RealEstateChatbot:
    """
    Factory function to create chatbot instance.
    
    Args:
        use_mock: If True, create mock chatbot for testing
        **kwargs: Arguments to pass to RealEstateChatbot
    
    Returns:
        Chatbot instance
    """
    if use_mock:
        return MockChatbot()
    else:
        return RealEstateChatbot(**kwargs)


if __name__ == '__main__':
    # Test the chatbot
    print("Testing Real Estate Chatbot...")
    
    # Use mock chatbot for testing
    chatbot = create_chatbot(use_mock=True)
    
    # Test questions
    test_questions = [
        "What is the forecast for property values?",
        "What are the main revenue risks?",
        "How confident are these predictions?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        response = chatbot.chat(question)
        print(f"A: {response}")
