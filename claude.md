# OHD LLM Analyzer - Claude Code Instructions

## Project Setup
Create a new Python module within your existing OHD repository structure:

```
template/
├── _data/
├── _includes/
├── assets/
└── llm_analyzer/          # New module
    ├── __init__.py
    ├── main.py
    ├── models/
    │   ├── __init__.py
    │   ├── llm_client.py
    │   └── structured_outputs.py
    ├── processors/
    │   ├── __init__.py
    │   ├── summarizer.py
    │   ├── evaluator.py
    │   ├── vocab_generator.py
    │   └── tagger.py
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    └── requirements.txt
```

## Core Implementation Tasks

### 1. Dependencies Setup
Create `requirements.txt`:
```
openai>=1.0.0
anthropic>=0.7.0
requests>=2.28.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pyyaml>=6.0
```

### 2. Configuration System (`config/settings.py`)
```python
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
```

### 3. Structured Output Models (`models/structured_outputs.py`)
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class QualityScore(BaseModel):
    score: int = Field(..., ge=0, le=1, description="1 if summary is good, 0 if not")
    reasoning: str = Field(..., description="Brief explanation for the score")

class ThematicTag(BaseModel):
    term: str = Field(..., description="Controlled vocabulary term")
    confidence: float = Field(..., ge=0.0, le=1.0)
    context: str = Field(..., description="Brief context where this theme appears")

class InterviewSummary(BaseModel):
    summary: str = Field(..., description="Concise summary of interview content")
    key_themes: List[str] = Field(..., description="3-5 main themes identified")
    notable_quotes: List[str] = Field(..., max_items=3)

class ControlledVocab(BaseModel):
    terms: List[str] = Field(..., description="List of thematic terms")
    definitions: Dict[str, str] = Field(..., description="Term definitions")
    hierarchies: Dict[str, List[str]] = Field(default_factory=dict, description="Parent-child relationships")

class SectionTags(BaseModel):
    section_id: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    content_preview: str = Field(..., max_length=200)
    tags: List[ThematicTag]
```

### 4. LLM Client (`models/llm_client.py`)
```python
import requests
import json
from typing import Dict, Any, Optional
from anthropic import Anthropic
from openai import OpenAI
from .structured_outputs import *
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
        
        return response_model.model_validate_json(response)
```

### 5. Processing Pipeline Components

#### Summarizer (`processors/summarizer.py`)
```python
from models.llm_client import LLMClient
from models.structured_outputs import InterviewSummary, QualityScore

class InterviewSummarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def summarize_interview(self, transcript: str, interview_metadata: dict) -> InterviewSummary:
        messages = [{
            "role": "user",
            "content": f"""
            Please create a summary of this oral history interview transcript:
            
            Interview Details:
            - Date: {interview_metadata.get('date', 'Unknown')}
            - Interviewer: {interview_metadata.get('interviewer', 'Unknown')}
            - Subject: {interview_metadata.get('subject', 'Unknown')}
            
            Transcript:
            {transcript[:8000]}  # Truncate for token limits
            
            Focus on key themes, significant events, and notable perspectives shared.
            """
        }]
        
        return self.llm.generate_structured(messages, InterviewSummary)
    
    def evaluate_summary_quality(self, original_transcript: str, summary: InterviewSummary) -> QualityScore:
        messages = [{
            "role": "user", 
            "content": f"""
            Evaluate if this summary adequately captures the key content of the original transcript:
            
            Original (first 2000 chars): {original_transcript[:2000]}
            
            Summary: {summary.summary}
            Key Themes: {', '.join(summary.key_themes)}
            
            Rate 1 if the summary is comprehensive and accurate, 0 if it needs improvement.
            """
        }]
        
        return self.llm.generate_structured(messages, QualityScore)
```

#### Vocabulary Generator (`processors/vocab_generator.py`)
```python
from typing import List
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
```

#### Section Tagger (`processors/tagger.py`)
```python
from typing import List
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
```

### 6. Main Pipeline (`main.py`)
```python
#!/usr/bin/env python3
import json
import os
import yaml
from pathlib import Path
from typing import List, Dict
from models.llm_client import LLMClient
from processors.summarizer import InterviewSummarizer
from processors.vocab_generator import VocabularyGenerator
from processors.tagger import SectionTagger

def load_interview_data(data_path: str) -> List[Dict]:
    """Load interview transcripts from _data folder"""
    interviews = []
    data_dir = Path(data_path)
    
    # Look for JSON and YAML files
    for file_path in data_dir.glob("*.{json,yml,yaml}"):
        with open(file_path, 'r') as f:
            if file_path.suffix == '.json':
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
            
            if isinstance(data, list):
                interviews.extend(data)
            else:
                interviews.append(data)
    
    return interviews

def main():
    # Initialize components
    llm_client = LLMClient()
    summarizer = InterviewSummarizer(llm_client)
    vocab_generator = VocabularyGenerator(llm_client)
    tagger = SectionTagger(llm_client)
    
    # Load interview data
    interviews = load_interview_data("_data")
    
    print(f"Processing {len(interviews)} interviews...")
    
    # Phase 1: Summarize and validate
    validated_summaries = []
    for interview in interviews:
        print(f"Summarizing interview: {interview.get('title', 'Unknown')}")
        
        summary = summarizer.summarize_interview(
            interview.get('transcript', ''),
            interview
        )
        
        # Evaluate summary quality  
        quality = summarizer.evaluate_summary_quality(
            interview.get('transcript', ''),
            summary
        )
        
        if quality.score == 0:
            print(f"Re-summarizing due to quality score: {quality.reasoning}")
            # Re-summarize with more specific instructions
            summary = summarizer.summarize_interview(
                interview.get('transcript', ''),
                interview
            )
        
        validated_summaries.append(summary)
    
    # Phase 2: Generate controlled vocabulary
    print("Generating controlled vocabulary...")
    vocab = vocab_generator.generate_controlled_vocab(validated_summaries)
    
    # Save vocabulary for Jekyll site
    vocab_output = {
        'terms': vocab.terms,
        'definitions': vocab.definitions,
        'hierarchies': vocab.hierarchies
    }
    
    with open('_data/controlled_vocab.json', 'w') as f:
        json.dump(vocab_output, f, indent=2)
    
    # Phase 3: Tag interview sections
    print("Tagging interview sections...")
    all_tags = []
    
    for i, interview in enumerate(interviews):
        transcript = interview.get('transcript', '')
        # Split into sections (you might want to customize this logic)
        sections = transcript.split('\n\n')  # Simple paragraph split
        
        interview_tags = []
        for j, section in enumerate(sections):
            if len(section.strip()) > 100:  # Skip very short sections
                section_id = f"interview_{i}_section_{j}"
                tags = tagger.tag_transcript_section(section, section_id, vocab)
                interview_tags.append(tags.dict())
        
        all_tags.append({
            'interview_id': interview.get('id', i),
            'title': interview.get('title', 'Unknown'),
            'sections': interview_tags
        })
    
    # Save tagged data for Jekyll
    with open('_data/interview_tags.json', 'w') as f:
        json.dump(all_tags, f, indent=2)
    
    print("Analysis complete! Check _data/ for generated vocabulary and tags.")

if __name__ == "__main__":
    main()
```

### 7. Environment Configuration
Create `.env` file in your project root:
```
# Choose your LLM provider
LLM_PROVIDER=ollama  # or 'claude' or 'openai'

# Ollama settings
OLLAMA_MODEL=llama2
OLLAMA_URL=http://localhost:11434

# Claude settings (if using)
ANTHROPIC_API_KEY=your_api_key_here

# OpenAI settings (if using)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

## Usage Instructions for Claude Code

1. **Setup the module structure:**
   ```bash
   mkdir -p llm_analyzer/{models,processors,config}
   touch llm_analyzer/__init__.py llm_analyzer/{models,processors,config}/__init__.py
   ```

2. **Install dependencies:**
   ```bash
   pip install -r llm_analyzer/requirements.txt
   ```

3. **Create the prompts configuration:**
   ```bash
   # Copy the prompts.yml content above to config/prompts.yml
   ```

4. **Configure your environment:**
   ```bash
   # Copy the .env template and configure for your LLM provider
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Tag a single CSV file:**
   ```bash
   # Quick method
   python llm_analyzer/tag_csv.py your_interview.csv
   
   # Or with more options
   python llm_analyzer/main.py your_interview.csv --output tagged_output.csv --save-vocab vocab.json
   ```

6. **Process all CSV files in _data directory:**
   ```bash
   cd llm_analyzer
   python main.py  # No arguments processes all CSVs in _data/
   ```

## CSV Processing Workflow

The system processes your CSV files in these steps:

1. **Extract Utterances** (100 rows at a time) - Identifies significant thematic content
2. **Generate Themes** - Creates thematic terms from utterances  
3. **Evaluate Quality** - Validates themes and regenerates if needed
4. **Build Vocabulary** - Creates controlled vocabulary from all themes
5. **Tag Segments** - Applies vocabulary terms to individual CSV rows
6. **Validate Tags** - Ensures tag quality before adding to CSV

**Output:** Your original CSV with a new `gen-tags` column containing the LLM-generated tags.

## Example Usage

For your CSV with structure:
```
speaker,words,tags,timestamp
Devin Becker,"And there we go with that...",,[0:00]
Rae Armantrout,"Mostly poetry.",,[01:43]
```

After processing:
```
speaker,words,tags,timestamp,gen-tags
Devin Becker,"And there we go with that...",,[0:00],
Rae Armantrout,"Mostly poetry.",,[01:43],"creative process; literary practice"
```

## Customizing for Your Content

**Modify prompts in `config/prompts.yml`** to:
- Adjust theme extraction for your specific oral history focus
- Add domain-specific vocabulary guidance
- Modify confidence thresholds and validation criteria

**Example prompt customization:**
```yaml
extract_utterances:
  user: |
    Focus specifically on utterances about:
    - Writing and creative processes  
    - Technology's impact on creative work
    - Professional development in the arts
    # ... rest of prompt
```

This approach breaks the analysis into manageable steps for smaller LLMs while maintaining quality through validation loops. Each step has a specific focus, making it easier to debug and refine the prompts for your particular oral history domain.