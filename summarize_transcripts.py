#!/usr/bin/env python3
"""
Focused pipeline for transcript summarization and controlled vocabulary generation.
Step 1: Create summaries of transcripts in small chunks
Step 2: Generate single controlled vocabulary (17-30 terms) from all summaries
"""
import json
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict
from simple_llm_client import SimpleLLMClient


class TranscriptSummarizer:
    def __init__(self, llm_client: SimpleLLMClient = None, config_path: str = "prompts.yml"):
        self.llm = llm_client or SimpleLLMClient()
        self.config = self.load_config(config_path)
    
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
                'segment_size': 20,
                'max_segment_chars': 2500,
                'target_vocab_terms': 25,
                'max_tokens': {
                    'segment_summary': 200,
                    'overall_summary': 300,
                    'vocabulary_generation': 800
                }
            }
        }
    
    def load_transcript(self, csv_path: Path) -> pd.DataFrame:
        """Load transcript CSV with basic cleaning"""
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows from {csv_path.name}")
        return df
    
    def extract_meaningful_segments(self, df: pd.DataFrame) -> List[str]:
        """Extract meaningful content segments for summarization"""
        segments = []
        meaningful_rows = []
        
        # Get configuration parameters
        segment_size = self.config['config'].get('segment_size', 20)
        max_segment_chars = self.config['config'].get('max_segment_chars', 2500)
        min_word_length = self.config['config'].get('min_word_length', 15)
        min_word_count = self.config['config'].get('min_word_count', 4)
        excluded_phrases = self.config['config'].get('excluded_phrases', ['Good.', 'OK.', 'Right.', 'Yes.', 'No.', 'Mm-hmm.'])
        
        # Filter for substantial content
        for _, row in df.iterrows():
            words = str(row.get('words', '')).strip()
            speaker = str(row.get('speaker', '')).strip()
            
            # Keep rows with substantial content
            if (len(words) > min_word_length and 
                len(words.split()) >= min_word_count and
                not words.startswith('[') and
                words not in excluded_phrases):
                
                meaningful_rows.append(f"{speaker}: {words}")
        
        # Group into segments to fit context windows
        for i in range(0, len(meaningful_rows), segment_size):
            segment = meaningful_rows[i:i+segment_size]
            if segment:  # Only add non-empty segments
                segment_text = "\n".join(segment)
                segments.append(segment_text[:max_segment_chars])  # Truncate very long segments
        
        print(f"    Created {len(segments)} segments from {len(meaningful_rows)} meaningful exchanges")
        return segments
    
    def summarize_segment(self, segment_text: str, transcript_name: str) -> str:
        """Summarize a single segment of transcript"""
        # Get prompt template from config
        prompt_template = self.config.get('prompts', {}).get('segment_summary', {}).get('template', '')
        if not prompt_template:
            # Fallback to default if not found in config
            prompt_template = """Summarize this segment from an oral history interview about writing and technology.

Interview: {transcript_name}

Transcript segment:
{segment_text}

Create a 2-3 sentence summary focusing on:
- Key themes about writing practices
- Technology's impact on creative work
- Professional development or career insights
- Changes in practice over time

Keep the summary concise but capture the main themes discussed."""
        
        prompt = prompt_template.format(transcript_name=transcript_name, segment_text=segment_text)
        max_tokens = self.config['config']['max_tokens'].get('segment_summary', 200)

        try:
            summary = self.llm.generate_text(prompt, max_tokens=max_tokens)
            return summary.strip()
        except Exception as e:
            print(f"    Warning: Error summarizing segment: {e}")
            return f"Segment from {transcript_name} covering writing and technology themes."
    
    def create_transcript_summary(self, csv_path: Path) -> Dict:
        """Create a comprehensive summary of an entire transcript"""
        print(f"\n📝 Summarizing {csv_path.name}...")
        
        # Load and segment transcript
        df = self.load_transcript(csv_path)
        segments = self.extract_meaningful_segments(df)
        
        if not segments:
            print(f"    ⚠️  No meaningful content found in {csv_path.name}")
            return {
                "filename": csv_path.name,
                "title": csv_path.stem,
                "segments_processed": 0,
                "segment_summaries": [],
                "overall_summary": f"Transcript {csv_path.stem} with minimal substantial content."
            }
        
        # Summarize each segment
        segment_summaries = []
        for i, segment in enumerate(segments, 1):
            print(f"    Processing segment {i}/{len(segments)}...")
            summary = self.summarize_segment(segment, csv_path.stem)
            segment_summaries.append(summary)
        
        # Create overall summary from segment summaries
        overall_summary = self.create_overall_summary(segment_summaries, csv_path.stem)
        
        return {
            "filename": csv_path.name,
            "title": csv_path.stem,
            "segments_processed": len(segments),
            "segment_summaries": segment_summaries,
            "overall_summary": overall_summary
        }
    
    def create_overall_summary(self, segment_summaries: List[str], transcript_name: str) -> str:
        """Create an overall summary from segment summaries"""
        if not segment_summaries:
            return f"Transcript {transcript_name} summary not available."
        
        # Combine segment summaries (limit for context window)
        combined_summaries = "\n\n".join(segment_summaries[:8])  # Limit to first 8 segments
        
        # Get prompt template from config
        prompt_template = self.config.get('prompts', {}).get('overall_summary', {}).get('template', '')
        if not prompt_template:
            # Fallback to default if not found in config
            prompt_template = """Create a comprehensive summary of this oral history interview based on these segment summaries.

Interview: {transcript_name}

Segment summaries:
{combined_summaries}

Create a 4-5 sentence overall summary that captures:
- The interviewee's main insights about writing and technology
- Key themes that emerge across the conversation
- Notable changes in practice or perspective mentioned
- The overall arc or focus of the interview

Focus on the most significant themes and insights."""
        
        prompt = prompt_template.format(transcript_name=transcript_name, combined_summaries=combined_summaries)
        max_tokens = self.config['config']['max_tokens'].get('overall_summary', 300)

        try:
            overall = self.llm.generate_text(prompt, max_tokens=max_tokens)
            return overall.strip()
        except Exception as e:
            print(f"    Warning: Error creating overall summary: {e}")
            # Fallback: use first segment summary
            return segment_summaries[0] if segment_summaries else f"Interview with themes about writing and technology."


class VocabularyGenerator:
    def __init__(self, llm_client: SimpleLLMClient = None, config: Dict = None):
        self.llm = llm_client or SimpleLLMClient()
        self.config = config
    
    def generate_controlled_vocabulary(self, summaries: List[Dict], target_terms: int = None) -> List[Dict]:
        """Generate controlled vocabulary from transcript summaries"""
        # Use config for target terms if not specified
        if target_terms is None:
            target_terms = self.config['config'].get('target_vocab_terms', 25) if self.config else 25
            
        print(f"\n🏷️  Generating controlled vocabulary ({target_terms} terms)...")
        
        # Combine all overall summaries
        all_summaries = []
        for summary_data in summaries:
            if summary_data.get('overall_summary'):
                title = summary_data.get('title', 'Unknown')
                overall = summary_data['overall_summary']
                all_summaries.append(f"[{title}]: {overall}")
        
        if not all_summaries:
            print("    ⚠️  No summaries available for vocabulary generation")
            return []
        
        # Combine summaries (limit for context window)
        combined_summaries = "\n\n".join(all_summaries[:10])  # Limit to prevent context overflow
        
        # Get prompt template from config
        prompt_template = self.config.get('prompts', {}).get('vocabulary_generation', {}).get('template', '') if self.config else ''
        if not prompt_template:
            # Fallback to default if not found in config
            prompt_template = """Based on these oral history interview summaries about writing and technology, create a controlled vocabulary of exactly {target_terms} terms for tagging and visualization.

Interview summaries:
{combined_summaries}

Create {target_terms} concise thematic terms (1-3 words each) that would be useful for:
- Filtering and segmenting the collection
- Visualizing themes across interviews
- Academic analysis of writing practices and technology

Focus on themes like:
- Writing processes and practices
- Technology's impact on creative work
- Professional development
- Changes over time
- Tools and methods
- Creative approaches

For each term, provide:
1. A short tag (1-3 words, lowercase, suitable for filtering)
2. A clear description (8-12 words explaining the theme)

Format your response exactly like this:
revision: iterative process of improving and refining written work
correspondence: communication with editors publishers and other writers  
technology: digital tools and their impact on writing practices
early: writing practices before widespread computer adoption

Provide exactly {target_terms} terms, focusing on the most important themes that appear across multiple interviews."""

        prompt = prompt_template.format(target_terms=target_terms, combined_summaries=combined_summaries)
        max_tokens = self.config['config']['max_tokens'].get('vocabulary_generation', 800) if self.config else 800

        try:
            response = self.llm.generate_text(prompt, max_tokens=max_tokens)
            vocabulary = self.parse_vocabulary_response(response, target_terms)
            
            if len(vocabulary) < target_terms * 0.7:  # If we got fewer than 70% of target
                print(f"    ⚠️  Got only {len(vocabulary)} terms, no fallback available")
                print(f"    💡 Consider adjusting prompts or trying again")
            
            print(f"    ✅ Generated {len(vocabulary)} vocabulary terms")
            return vocabulary
            
        except Exception as e:
            print(f"    ❌ Error generating vocabulary: {e}")
            print(f"    💡 Consider checking LLM connection or adjusting prompts")
            return []
    
    def parse_vocabulary_response(self, response: str, target_terms: int) -> List[Dict]:
        """Parse vocabulary terms from LLM response"""
        vocabulary = []
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for "term: description" format
            if ':' in line and len(line.split(':')) == 2:
                parts = line.split(':', 1)
                raw_tag = parts[0].strip().lower()
                description = parts[1].strip()
                
                # Clean numerical prefixes from tags (e.g., "1. revision" -> "revision")
                import re
                tag = re.sub(r'^\d+\.\s*', '', raw_tag).strip()
                
                # Validate tag format
                if tag and description and len(tag.split()) <= 3 and len(tag) > 1:
                    vocabulary.append({
                        "tag": tag,
                        "description": description
                    })
                    
                    if len(vocabulary) >= target_terms:
                        break
        
        return vocabulary[:target_terms]  # Ensure we don't exceed target
    
    def save_vocabulary_csv(self, vocabulary: List[Dict], output_path: str):
        """Save vocabulary as gen-filters.csv"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            f.write("tag,description\n")
            for item in vocabulary:
                tag = item['tag']
                description = item['description']
                f.write(f'"{tag}","{description}"\n')
        
        print(f"    💾 Saved vocabulary to {output_path}")


def main():
    print("🎙️  OHD Transcript Summarizer & Vocabulary Generator")
    print("=" * 60)
    
    # Initialize components
    print("🤖 Initializing LLM client...")
    llm_client = SimpleLLMClient()
    
    if not llm_client.test_connection():
        print("❌ LLM client connection failed")
        return
    
    print(f"✅ Connected to {llm_client.provider} ({llm_client.model})")
    
    # Find transcript files
    transcripts_dir = Path("_data/transcripts")
    if not transcripts_dir.exists():
        print(f"❌ Directory {transcripts_dir} does not exist")
        return
    
    csv_files = list(transcripts_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {transcripts_dir}")
        return
    
    print(f"📁 Found {len(csv_files)} transcript files")
    
    # Step 1: Create summaries
    summarizer = TranscriptSummarizer(llm_client)
    all_summaries = []
    
    for csv_file in csv_files:
        try:
            summary_data = summarizer.create_transcript_summary(csv_file)
            all_summaries.append(summary_data)
        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")
            continue
    
    if not all_summaries:
        print("❌ No summaries created")
        return
    
    # Save summaries.json
    summaries_path = transcripts_dir / "summaries.json"
    with open(summaries_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved summaries to {summaries_path}")
    print(f"    📊 {len(all_summaries)} transcripts summarized")
    
    # Step 2: Generate controlled vocabulary
    vocab_generator = VocabularyGenerator(llm_client, summarizer.config)
    vocabulary = vocab_generator.generate_controlled_vocabulary(all_summaries)
    
    if vocabulary:
        # Save gen-filters.csv
        gen_filters_path = Path("_data/gen-filters.csv")
        gen_filters_path.parent.mkdir(exist_ok=True)
        vocab_generator.save_vocabulary_csv(vocabulary, str(gen_filters_path))
        
        print(f"\n🎉 Pipeline complete!")
        print(f"📁 Output files:")
        print(f"   📝 Summaries: {summaries_path}")
        print(f"   🏷️  Vocabulary: {gen_filters_path} ({len(vocabulary)} terms)")
        
        # Show sample vocabulary
        print(f"\n📋 Sample vocabulary terms:")
        for i, item in enumerate(vocabulary[:5]):
            print(f"   {i+1}. {item['tag']}: {item['description']}")
        if len(vocabulary) > 5:
            print(f"   ... and {len(vocabulary) - 5} more terms")
    else:
        print("❌ No vocabulary generated")


if __name__ == "__main__":
    main()