import csv
import pandas as pd
import sys
from typing import List, Dict, Tuple
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.llm_client import LLMClient
from models.structured_outputs import (
    UtteranceGroup, ThematicTerms, GeneratedFilters, 
    TranscriptTags, RowTags, QualityValidation
)
from config.prompts import PromptsManager

class CSVProcessor:
    def __init__(self, llm_client: LLMClient, prompts_manager: PromptsManager = None):
        self.llm = llm_client
        self.prompts = prompts_manager or PromptsManager()
    
    def load_transcript_csv(self, file_path: Path) -> pd.DataFrame:
        """Load a transcript CSV file"""
        return pd.read_csv(file_path)
    
    def extract_utterances(self, transcript_df: pd.DataFrame, batch_size: int = None) -> List[UtteranceGroup]:
        """Extract significant utterances from transcript in batches"""
        if batch_size is None:
            batch_size = self.prompts.get_batch_size()
            
        utterance_groups = []
        
        # Process rows in batches
        for i in range(0, len(transcript_df), batch_size):
            batch = transcript_df.iloc[i:i+batch_size]
            
            # Create text from batch
            batch_text = ""
            for _, row in batch.iterrows():
                if pd.notna(row['words']) and row['words'].strip():
                    speaker = row.get('speaker', 'Unknown')
                    words = row['words']
                    batch_text += f"{speaker}: {words}\n"
            
            if not batch_text.strip():
                continue
                
            messages = self.prompts.get_message(
                'extract_utterances',
                batch_text=batch_text[:2000]
            )
            
            try:
                utterance_group = self.llm.generate_structured(messages, UtteranceGroup)
                if utterance_group.utterances:
                    utterance_groups.append(utterance_group)
                else:
                    print(f"    ⚠️  No utterances extracted from batch {i}-{i+batch_size}")
            except Exception as e:
                print(f"    ❌ Error extracting utterances from batch {i}-{i+batch_size}: {e}")
                # Try to create a fallback with empty utterances if batch has content
                if batch_text.strip():
                    print(f"    📝 Creating fallback entry for batch {i}-{i+batch_size}")
                    fallback = UtteranceGroup(
                        utterances=[],
                        context=f"Processing batch {i}-{i+batch_size} (LLM parsing failed)"
                    )
                    utterance_groups.append(fallback)
                continue
        
        return utterance_groups
    
    def generate_thematic_terms(self, utterance_groups: List[UtteranceGroup]) -> ThematicTerms:
        """Generate thematic terms from utterances"""
        all_utterances = []
        for group in utterance_groups:
            all_utterances.extend(group.utterances)
        
        messages = self.prompts.get_message(
            'generate_thematic_terms',
            utterances_text=chr(10).join(all_utterances[:50])
        )
        
        return self.llm.generate_structured(messages, ThematicTerms)
    
    def create_controlled_vocabulary(self, terms: List[str]) -> GeneratedFilters:
        """Create controlled vocabulary similar to filters.csv"""
        messages = self.prompts.get_message(
            'create_controlled_vocabulary',
            terms=', '.join(terms)
        )
        
        return self.llm.generate_structured(messages, GeneratedFilters)
    
    def tag_transcript_rows(self, transcript_df: pd.DataFrame, vocabulary: GeneratedFilters, filename: str) -> TranscriptTags:
        """Tag individual rows in transcript with relevant terms"""
        available_tags = [f.tag for f in vocabulary.filters]
        tag_descriptions = {f.tag: f.description for f in vocabulary.filters}
        
        row_tags = []
        
        # Process rows in small batches for efficiency
        for i in range(0, len(transcript_df), 10):
            batch = transcript_df.iloc[i:i+10]
            
            batch_data = []
            for idx, row in batch.iterrows():
                if pd.notna(row['words']) and row['words'].strip():
                    batch_data.append({
                        'index': idx,
                        'speaker': row.get('speaker', ''),
                        'words': row['words'][:500],  # Limit for token efficiency
                        'timestamp': row.get('timestamp', '')
                    })
            
            if not batch_data:
                continue
            
            batch_text = "\n".join([
                f"Row {item['index']}: {item['speaker']}: {item['words']}"
                for item in batch_data
            ])
            
            messages = self.prompts.get_message(
                'tag_transcript_rows',
                tag_descriptions=chr(10).join([f"- {tag}: {desc}" for tag, desc in tag_descriptions.items()]),
                batch_text=batch_text
            )
            
            try:
                batch_tags = self.llm.generate_structured(messages, TranscriptTags)
                row_tags.extend(batch_tags.row_tags)
            except Exception as e:
                print(f"Error tagging batch starting at row {i}: {e}")
                # Add empty tags for this batch
                for item in batch_data:
                    row_tags.append(RowTags(row_index=item['index'], gen_tags=""))
        
        return TranscriptTags(filename=filename, row_tags=row_tags)
    
    def validate_tags_quality(self, original_rows: List[str], generated_tags: List[str]) -> QualityValidation:
        """Validate quality of generated tags"""
        sample_rows = original_rows[:5]  # Sample for validation
        sample_tags = generated_tags[:5]
        
        samples_text = chr(10).join([f"Row: {row}\nTags: {tags}\n" for row, tags in zip(sample_rows, sample_tags)])
        
        messages = self.prompts.get_message(
            'validate_tags_quality',
            samples=samples_text
        )
        
        return self.llm.generate_structured(messages, QualityValidation)
    
    def save_transcript_with_tags(self, transcript_df: pd.DataFrame, tags: TranscriptTags, output_path: Path):
        """Save transcript CSV with new gen-tags column"""
        # Create tags lookup
        tags_lookup = {tag.row_index: tag.gen_tags for tag in tags.row_tags}
        
        # Add gen-tags column
        transcript_df['gen-tags'] = transcript_df.index.map(lambda x: tags_lookup.get(x, ""))
        
        # Save to new file
        transcript_df.to_csv(output_path, index=False)
    
    def save_generated_filters(self, filters: GeneratedFilters, output_path: Path):
        """Save generated filters to gen-filters.csv"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['tag', 'description'])
            
            for filter_entry in filters.filters:
                writer.writerow([filter_entry.tag, filter_entry.description])