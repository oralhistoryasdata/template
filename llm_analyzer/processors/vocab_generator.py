import sys
from pathlib import Path
from typing import List

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.llm_client import LLMClient
from models.structured_outputs import InterviewSummary, ControlledVocab

class VocabularyGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def generate_controlled_vocab(self, summaries: List[InterviewSummary]) -> ControlledVocab:
        # Combine all themes from summaries
        all_themes = []
        for summary in summaries:
            all_themes.extend(summary.key_themes)
        
        messages = [{
            "role": "user",
            "content": f"""
            Based on these themes from oral history interviews, create a controlled vocabulary:
            
            Raw themes: {', '.join(all_themes)}
            
            Create standardized terms that:
            1. Consolidate similar concepts
            2. Use consistent terminology
            3. Include broader category relationships
            4. Provide clear definitions
            
            Focus on themes relevant to oral history analysis like: personal experiences, 
            historical events, social contexts, cultural practices, etc.
            """
        }]
        
        return self.llm.generate_structured(messages, ControlledVocab)