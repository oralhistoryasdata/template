import requests
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from anthropic import Anthropic
from openai import OpenAI

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.structured_outputs import *
from config.settings import LLMConfig, get_config

class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config()
        self._setup_client()
    
    def _setup_client(self):
        if self.config.provider == 'claude':
            self.client = Anthropic(api_key=self.config.api_key)
        elif self.config.provider == 'openai':
            self.client = OpenAI(api_key=self.config.api_key)
        elif self.config.provider == 'ollama':
            self.client = None  # Use requests directly
    
    def _call_ollama(self, messages: List[Dict], response_format: Optional[Dict] = None) -> str:
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.config.temperature}
        }
        
        if response_format:
            payload["format"] = "json"
        
        response = requests.post(
            f"{self.config.base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()["message"]["content"]
    
    def generate_structured(self, messages: List[Dict], response_model: BaseModel) -> BaseModel:
        """Generate structured output using the configured LLM"""
        
        # Add JSON schema instruction to the last message
        schema_prompt = f"\n\nPlease respond with valid JSON matching this schema:\n{response_model.model_json_schema()}"
        messages[-1]["content"] += schema_prompt
        
        if self.config.provider == 'ollama':
            response = self._call_ollama(messages, {"type": "json_object"})
        elif self.config.provider == 'claude':
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages
            ).content[0].text
        elif self.config.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                response_format={"type": "json_object"}
            ).choices[0].message.content
        
        try:
            return response_model.model_validate_json(response)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response[:500]}...")
            raise