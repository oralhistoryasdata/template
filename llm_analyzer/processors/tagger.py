import sys
from pathlib import Path
from typing import List

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.llm_client import LLMClient
from models.structured_outputs import ControlledVocab, SectionTags

class SectionTagger:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def tag_transcript_section(self, section_text: str, section_id: str, vocab: ControlledVocab) -> SectionTags:
        vocab_terms = ", ".join(vocab.terms)
        
        messages = [{
            "role": "user",
            "content": f"""
            Tag this interview section with relevant terms from the controlled vocabulary:
            
            Available terms: {vocab_terms}
            
            Section text:
            {section_text}
            
            Select only terms that clearly apply to this section's content. 
            Include confidence scores and brief context for each tag.
            """
        }]
        
        return self.llm.generate_structured(messages, SectionTags)