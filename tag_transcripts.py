#!/usr/bin/env python3
"""
Phase 1: Row-level transcript tagging with thematic section detection.

This script processes CSV transcripts to identify thematic sections and apply 
controlled vocabulary tags. It prioritizes coherent thematic groupings over 
individual row analysis.
"""
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from simple_llm_client import SimpleLLMClient


class ThematicTagger:
    def __init__(self, llm_client: SimpleLLMClient = None, config_path: str = "prompts.yml"):
        self.llm = llm_client or SimpleLLMClient()
        self.config = self.load_config(config_path)
        self.vocabulary = self.load_vocabulary()
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from prompts.yml"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Default configuration if prompts.yml is not available"""
        return {
            'config': {
                'thematic_section_size': 8,
                'overlap_rows': 2,
                'min_section_length': 100,
                'max_tokens': {'section_analysis': 400, 'row_tagging': 200}
            }
        }
    
    def load_vocabulary(self) -> List[Dict]:
        """Load generated vocabulary from gen-filters.csv"""
        vocab_path = Path("_data/gen-filters.csv")
        if not vocab_path.exists():
            print(f"Warning: {vocab_path} not found. Run summarize_transcripts.py first.")
            return []
        
        try:
            df = pd.read_csv(vocab_path)
            vocab_list = []
            for _, row in df.iterrows():
                vocab_list.append({
                    'tag': row['tag'],
                    'description': row['description']
                })
            print(f"✅ Loaded {len(vocab_list)} vocabulary terms from {vocab_path}")
            return vocab_list
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            return []
    
    def create_thematic_sections(self, df: pd.DataFrame) -> List[Dict]:
        """
        Group CSV rows into thematic sections for coherent tagging.
        Uses sliding window approach to identify natural topic boundaries.
        """
        section_size = self.config['config'].get('thematic_section_size', 8)
        overlap = self.config['config'].get('overlap_rows', 2)
        min_length = self.config['config'].get('min_section_length', 100)
        
        sections = []
        
        # Filter for meaningful content rows
        meaningful_rows = []
        for idx, row in df.iterrows():
            words = str(row.get('words', '')).strip()
            if len(words) > min_length//10 and len(words.split()) >= 3:  # Basic content filter
                meaningful_rows.append({
                    'original_index': idx,
                    'speaker': str(row.get('speaker', '')),
                    'words': words,
                    'timestamp': str(row.get('timestamp', '')),
                    'existing_tags': str(row.get('tags', ''))
                })
        
        if not meaningful_rows:
            print("No meaningful content found for sectioning")
            return []
        
        # Create overlapping sections
        for i in range(0, len(meaningful_rows), section_size - overlap):
            section_rows = meaningful_rows[i:i + section_size]
            if not section_rows:
                break
            
            # Calculate section text length
            section_text = "\n".join([f"{r['speaker']}: {r['words']}" for r in section_rows])
            
            if len(section_text) >= min_length:  # Only include substantial sections
                sections.append({
                    'section_id': f"section_{i//section_size + 1}",
                    'start_row': section_rows[0]['original_index'],
                    'end_row': section_rows[-1]['original_index'],
                    'rows': section_rows,
                    'section_text': section_text[:2000]  # Truncate for context limits
                })
        
        print(f"📝 Created {len(sections)} thematic sections from {len(meaningful_rows)} meaningful rows")
        return sections
    
    def analyze_section_themes(self, section: Dict) -> Dict:
        """
        Analyze a thematic section to identify primary themes and 
        determine which vocabulary terms apply.
        """
        if not self.vocabulary:
            return {'themes': [], 'confidence': 0.0, 'reasoning': 'No vocabulary available'}
        
        # Create vocabulary reference string
        vocab_terms = []
        for v in self.vocabulary:
            vocab_terms.append(f"• {v['tag']}: {v['description']}")
        vocab_text = "\n".join(vocab_terms)
        
        # Get prompt template from config
        prompt_template = self.config.get('prompts', {}).get('section_analysis', {}).get('template', '')
        if not prompt_template:
            # Fallback to default if not found in config
            prompt_template = """Analyze this section of an oral history interview about writing and technology to identify the main themes.

AVAILABLE VOCABULARY TERMS:
{vocab_terms}

INTERVIEW SECTION:
{section_text}

Your task:
1. Identify the PRIMARY THEMES in this section (1-4 themes maximum)
2. Select ONLY the vocabulary terms that clearly and strongly apply
3. Focus on themes that span multiple exchanges in this section

Respond in this exact format:
THEMES: [list only the tag names that apply, separated by semicolons]
CONFIDENCE: [0.0-1.0 score for how well these themes capture the section]
REASONING: [Brief explanation of why these themes were selected]

Example:
THEMES: technology; revision; collaboration
CONFIDENCE: 0.8
REASONING: Section discusses computer adoption for writing, editing practices, and working with others

Be selective - only choose themes that are clearly evident and central to the section."""
        
        prompt = prompt_template.format(vocab_terms=vocab_text, section_text=section['section_text'])

        try:
            response = self.llm.generate_text(prompt, max_tokens=self.config['config']['max_tokens'].get('section_analysis', 400))
            return self.parse_section_analysis(response, section['section_id'])
        except Exception as e:
            print(f"    Error analyzing section {section['section_id']}: {e}")
            return {'themes': [], 'confidence': 0.0, 'reasoning': f'Analysis failed: {e}'}
    
    def parse_section_analysis(self, response: str, section_id: str) -> Dict:
        """Parse the LLM response for section theme analysis"""
        themes = []
        confidence = 0.0
        reasoning = "Could not parse response"
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('THEMES:'):
                themes_text = line.replace('THEMES:', '').strip()
                if themes_text and themes_text.lower() != 'none':
                    themes = [t.strip() for t in themes_text.split(';') if t.strip()]
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except ValueError:
                    confidence = 0.0
            
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        # Validate themes against vocabulary
        valid_themes = []
        valid_tags = [v['tag'] for v in self.vocabulary]
        for theme in themes:
            if theme in valid_tags:
                valid_themes.append(theme)
        
        return {
            'themes': valid_themes,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def apply_section_tags_to_rows(self, section: Dict, section_analysis: Dict) -> List[Dict]:
        """
        Apply the section's thematic tags to individual rows,
        with some rows potentially getting additional specific tags.
        """
        section_themes = section_analysis['themes']
        if not section_themes:
            # No themes identified for this section
            return [{'row_index': row['original_index'], 'tags': ''} for row in section['rows']]
        
        row_tags = []
        
        # Apply section themes to all rows in the section
        for row in section['rows']:
            # Start with section-level themes
            tags = section_themes.copy()
            
            # Check if this specific row has particularly strong connection to additional themes
            additional_tags = self.analyze_row_specificity(row, section_themes)
            if additional_tags:
                # Add unique additional tags
                for tag in additional_tags:
                    if tag not in tags:
                        tags.append(tag)
            
            # Format tags as semicolon-separated string
            tags_string = '; '.join(tags) if tags else ''
            
            row_tags.append({
                'row_index': row['original_index'],
                'tags': tags_string
            })
        
        return row_tags
    
    def analyze_row_specificity(self, row: Dict, section_themes: List[str]) -> List[str]:
        """
        Check if a specific row has strong connections to themes beyond 
        the section themes (lightweight analysis).
        """
        # For now, keep this simple - primarily rely on section-level themes
        # This could be expanded later for more granular analysis
        return []
    
    def tag_transcript(self, csv_path: Path) -> bool:
        """
        Tag an entire transcript CSV file with thematic vocabulary terms.
        Creates sections, analyzes themes, and applies tags to rows.
        """
        print(f"\n🏷️  Tagging transcript: {csv_path.name}")
        
        try:
            # Load transcript
            df = pd.read_csv(csv_path)
            print(f"    📄 Loaded {len(df)} rows")
            
            if len(df) == 0:
                print("    ⚠️  Empty transcript, skipping")
                return False
            
            # Create thematic sections
            sections = self.create_thematic_sections(df)
            if not sections:
                print("    ⚠️  No thematic sections created, skipping")
                return False
            
            # Analyze each section for themes
            section_analyses = []
            for i, section in enumerate(sections, 1):
                print(f"    🔍 Analyzing section {i}/{len(sections)}: rows {section['start_row']}-{section['end_row']}")
                analysis = self.analyze_section_themes(section)
                section_analyses.append(analysis)
                
                if analysis['themes']:
                    themes_str = ', '.join(analysis['themes'])
                    print(f"        ✅ Themes: {themes_str} (confidence: {analysis['confidence']:.2f})")
                else:
                    print(f"        ❌ No themes identified")
            
            # Apply section-level tags to individual rows
            all_row_tags = {}
            for section, analysis in zip(sections, section_analyses):
                row_tags = self.apply_section_tags_to_rows(section, analysis)
                for tag_info in row_tags:
                    all_row_tags[tag_info['row_index']] = tag_info['tags']
            
            # Add gen-tags column to original dataframe
            df['gen-tags'] = ''
            for row_idx, tags in all_row_tags.items():
                df.at[row_idx, 'gen-tags'] = tags
            
            # Save tagged CSV
            output_path = csv_path.parent / f"{csv_path.stem}_tagged.csv"
            df.to_csv(output_path, index=False)
            
            # Report results
            tagged_rows = len([tags for tags in all_row_tags.values() if tags])
            print(f"    💾 Saved to {output_path}")
            print(f"    📊 Tagged {tagged_rows}/{len(df)} rows ({tagged_rows/len(df)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"    ❌ Error tagging {csv_path.name}: {e}")
            return False


def main():
    print("🏷️  OHD Transcript Tagger (Phase 1: Thematic Sections)")
    print("=" * 60)
    
    # Initialize components
    print("🤖 Initializing LLM client...")
    llm_client = SimpleLLMClient()
    
    if not llm_client.test_connection():
        print("❌ LLM client connection failed")
        return
    
    print(f"✅ Connected to {llm_client.provider} ({llm_client.model})")
    
    # Initialize tagger
    tagger = ThematicTagger(llm_client)
    
    if not tagger.vocabulary:
        print("❌ No vocabulary loaded. Please run summarize_transcripts.py first.")
        return
    
    # Find transcript files
    transcripts_dir = Path("_data/transcripts")
    if not transcripts_dir.exists():
        print(f"❌ Directory {transcripts_dir} does not exist")
        return
    
    csv_files = list(transcripts_dir.glob("*.csv"))
    # Exclude already tagged files
    csv_files = [f for f in csv_files if not f.name.endswith('_tagged.csv')]
    
    if not csv_files:
        print(f"❌ No untagged CSV files found in {transcripts_dir}")
        return
    
    print(f"📁 Found {len(csv_files)} transcript files to tag")
    
    # Process each transcript
    successful_tags = 0
    for csv_file in csv_files:
        if tagger.tag_transcript(csv_file):
            successful_tags += 1
    
    print(f"\n🎉 Tagging complete!")
    print(f"✅ Successfully tagged {successful_tags}/{len(csv_files)} transcripts")
    
    if successful_tags > 0:
        print(f"\n📁 Tagged files saved as *_tagged.csv in {transcripts_dir}")
        print("🔍 Check the gen-tags column for applied thematic vocabulary")


if __name__ == "__main__":
    main()