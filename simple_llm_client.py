#!/usr/bin/env python3
"""
Simplified LLM client without Pydantic dependencies.
Returns plain text responses for easier parsing.
"""
import requests
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleLLMClient:
    def __init__(self):
        self.provider = os.getenv('LLM_PROVIDER', 'ollama').lower()
        self.setup_client()
    
    def setup_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == 'ollama':
            self.base_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
            self.model = os.getenv('OLLAMA_MODEL', 'llama2')
            self.client = None
        elif self.provider == 'claude':
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                self.model = 'claude-3-haiku-20240307'  # Use faster model for processing
            except ImportError:
                print("Warning: anthropic package not installed. Install with: pip install anthropic")
                self.fallback_to_ollama()
        elif self.provider == 'openai':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
                self.fallback_to_ollama()
        else:
            self.fallback_to_ollama()
    
    def fallback_to_ollama(self):
        """Fallback to Ollama if other providers fail"""
        print("Falling back to Ollama...")
        self.provider = 'ollama'
        self.base_url = 'http://localhost:11434'
        self.model = 'llama2'
        self.client = None
    
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate plain text response from LLM"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            if self.provider == 'ollama':
                return self._call_ollama(messages)
            elif self.provider == 'claude':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    messages=messages
                )
                return response.content[0].text
            elif self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error with {self.provider}: {e}")
            if self.provider != 'ollama':
                print("Attempting fallback to Ollama...")
                self.fallback_to_ollama()
                return self._call_ollama(messages)
            raise
    
    def _call_ollama(self, messages: List[Dict]) -> str:
        """Call Ollama API directly"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama connection error: {e}")

    def test_connection(self) -> bool:
        """Test if the LLM client is working"""
        try:
            response = self.generate_text("Say 'Hello' if you can hear me.", max_tokens=50)
            return "hello" in response.lower()
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

if __name__ == "__main__":
    # Test the client
    client = SimpleLLMClient()
    print(f"Using provider: {client.provider}")
    print(f"Using model: {client.model}")
    
    if client.test_connection():
        print("✅ LLM client working!")
    else:
        print("❌ LLM client connection failed")