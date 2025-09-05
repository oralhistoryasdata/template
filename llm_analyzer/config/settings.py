from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    provider: str  # 'ollama', 'claude', 'openai'
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000

def get_config() -> LLMConfig:
    provider = os.getenv('LLM_PROVIDER', 'ollama').lower()
    
    if provider == 'ollama':
        return LLMConfig(
            provider='ollama',
            model_name=os.getenv('OLLAMA_MODEL', 'llama2'),
            base_url=os.getenv('OLLAMA_URL', 'http://localhost:11434'),
        )
    elif provider == 'claude':
        return LLMConfig(
            provider='claude',
            model_name='claude-3-sonnet-20240229',
            api_key=os.getenv('ANTHROPIC_API_KEY'),
        )
    elif provider == 'openai':
        return LLMConfig(
            provider='openai', 
            model_name=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            api_key=os.getenv('OPENAI_API_KEY'),
        )