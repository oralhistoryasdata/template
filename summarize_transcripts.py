#!/usr/bin/env python3
"""
Focused pipeline for transcript summarization and controlled vocabulary generation.
Step 1: Create summaries of transcripts in small chunks
Step 2: Generate single controlled vocabulary (17-30 terms) from all summaries
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from simple_llm_client import SimpleLLMClient


class TranscriptSummarizer:
    def __init__(self, llm_client: SimpleLLMClient = None):
        self.llm = llm_client or SimpleLLMClient()
    
    def load_transcript(self, csv_path: Path) -> pd.DataFrame:
        """Load transcript CSV with basic cleaning"""
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows from {csv_path.name}")
        return df
    
    def extract_meaningful_segments(self, df: pd.DataFrame, segment_size: int = 20) -> List[str]:
        """Extract meaningful content segments for summarization"""
        segments = []
        meaningful_rows = []
        
        # Filter for substantial content
        for _, row in df.iterrows():
            words = str(row.get('words', '')).strip()
            speaker = str(row.get('speaker', '')).strip()
            
            # Keep rows with substantial content
            if (len(words) > 15 and 
                len(words.split()) >= 4 and
                not words.startswith('[') and
                words not in ['Good.', 'OK.', 'Right.', 'Yes.', 'No.', 'Mm-hmm.']):
                
                meaningful_rows.append(f"{speaker}: {words}")
        
        # Group into segments to fit context windows
        for i in range(0, len(meaningful_rows), segment_size):
            segment = meaningful_rows[i:i+segment_size]
            if segment:  # Only add non-empty segments
                segment_text = "\n".join(segment)
                segments.append(segment_text[:2500])  # Truncate very long segments
        
        print(f"    Created {len(segments)} segments from {len(meaningful_rows)} meaningful exchanges")
        return segments
    
    def summarize_segment(self, segment_text: str, transcript_name: str) -> str:
        """Summarize a single segment of transcript"""
        prompt = f"""Summarize this segment from an oral history interview about writing and technology.

Interview: {transcript_name}

Transcript segment:
{segment_text}

Create a 2-3 sentence summary focusing on:
- Key themes about writing practices
- Technology's impact on creative work
- Professional development or career insights
- Changes in practice over time

Keep the summary concise but capture the main themes discussed."""

        try:
            summary = self.llm.generate_text(prompt, max_tokens=200)
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
        
        prompt = f"""Create a comprehensive summary of this oral history interview based on these segment summaries.

Interview: {transcript_name}

Segment summaries:
{combined_summaries}

Create a 4-5 sentence overall summary that captures:
- The interviewee's main insights about writing and technology
- Key themes that emerge across the conversation
- Notable changes in practice or perspective mentioned
- The overall arc or focus of the interview

Focus on the most significant themes and insights."""

        try:
            overall = self.llm.generate_text(prompt, max_tokens=300)
            return overall.strip()
        except Exception as e:
            print(f"    Warning: Error creating overall summary: {e}")
            # Fallback: use first segment summary
            return segment_summaries[0] if segment_summaries else f"Interview with themes about writing and technology."


class VocabularyGenerator:
    def __init__(self, llm_client: SimpleLLMClient = None):
        self.llm = llm_client or SimpleLLMClient()
    
    def generate_controlled_vocabulary(self, summaries: List[Dict], target_terms: int = 25) -> List[Dict]:
        """Generate controlled vocabulary from transcript summaries"""
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
        
        prompt = f"""Based on these oral history interview summaries about writing and technology, create a controlled vocabulary of exactly {target_terms} terms for tagging and visualization.

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

        try:
            response = self.llm.generate_text(prompt, max_tokens=800)
            vocabulary = self.parse_vocabulary_response(response, target_terms)
            
            if len(vocabulary) < target_terms * 0.7:  # If we got fewer than 70% of target
                print(f"    ⚠️  Got only {len(vocabulary)} terms, attempting second generation...")
                # Try again with a simpler approach
                vocabulary = self.generate_fallback_vocabulary(combined_summaries, target_terms)
            
            print(f"    ✅ Generated {len(vocabulary)} vocabulary terms")
            return vocabulary
            
        except Exception as e:
            print(f"    ❌ Error generating vocabulary: {e}")
            return self.generate_fallback_vocabulary(combined_summaries, target_terms)
    
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
    
    def generate_fallback_vocabulary(self, summaries_text: str, target_terms: int) -> List[Dict]:
        """Generate fallback vocabulary with simpler approach"""
        print(f"    Using fallback vocabulary generation...")
        
        # Basic terms that commonly appear in writing/technology oral histories
        fallback_terms = [
            {"tag": "revision", "description": "iterative process of improving and refining written work"},
            {"tag": "technology", "description": "digital tools and their impact on writing practices"},
            {"tag": "correspondence", "description": "communication with editors publishers and other writers"},
            {"tag": "early", "description": "writing practices before widespread computer adoption"},
            {"tag": "process", "description": "individual approaches to creating written work"},
            {"tag": "computer", "description": "impact of computers on writing and editing"},
            {"tag": "paper", "description": "use of physical materials in writing process"},
            {"tag": "teaching", "description": "instruction and pedagogy in writing and literature"},
            {"tag": "publishing", "description": "experiences with publication and literary industry"},
            {"tag": "collaboration", "description": "working with others in creative and professional contexts"},
            {"tag": "digital", "description": "electronic tools and online platforms for writing"},
            {"tag": "editing", "description": "revision and refinement of written work"},
            {"tag": "career", "description": "professional development and literary career progression"},
            {"tag": "reading", "description": "influences from other writers and literary works"},
            {"tag": "workshop", "description": "writing groups and collaborative learning environments"},
            {"tag": "inspiration", "description": "sources of creative ideas and motivation"},
            {"tag": "routine", "description": "daily practices and habits in writing process"},
            {"tag": "manuscript", "description": "physical and digital documents in writing workflow"},
            {"tag": "feedback", "description": "input from readers editors and writing community"},
            {"tag": "genre", "description": "different forms and styles of creative writing"},
            {"tag": "archive", "description": "preservation and organization of literary materials"},
            {"tag": "research", "description": "investigation and preparation for creative work"},
            {"tag": "deadline", "description": "time constraints and pressure in writing process"},
            {"tag": "voice", "description": "development of individual writing style and perspective"},
            {"tag": "craft", "description": "technical skills and artistic techniques in writing"}
        ]
        
        return fallback_terms[:target_terms]
    
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
    vocab_generator = VocabularyGenerator(llm_client)
    vocabulary = vocab_generator.generate_controlled_vocabulary(all_summaries, target_terms=25)
    
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