import yaml
from pathlib import Path
from typing import Dict, Any

class PromptsManager:
    def __init__(self, prompts_file: str = None):
        if prompts_file is None:
            prompts_file = Path(__file__).parent / "prompts.yml"
        
        with open(prompts_file, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
        
        self.config = self.data.get('config', {})
        self.prompts = self.data.get('prompts', {})
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_batch_size(self) -> int:
        """Get the batch size for processing rows"""
        return self.get_config('batch_size', 5)
    
    def get_vocab_terms_range(self) -> str:
        """Get the vocabulary terms range"""
        return self.get_config('vocab_terms_range', '3-6')
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get formatted prompt with variables filled in"""
        if prompt_name not in self.prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found in prompts.yml")
        
        prompt_data = self.prompts[prompt_name]
        if 'user' not in prompt_data:
            raise KeyError(f"Prompt '{prompt_name}' missing 'user' key")
        
        # Add config values to kwargs for template substitution
        template_vars = {
            'vocab_terms_range': self.get_vocab_terms_range(),
            **kwargs
        }
        
        return prompt_data['user'].format(**template_vars)
    
    def get_message(self, prompt_name: str, **kwargs) -> list:
        """Get formatted message for LLM client"""
        return [{
            "role": "user",
            "content": self.get_prompt(prompt_name, **kwargs)
        }]