#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List

# Add llm_analyzer to the path
sys.path.append(str(Path(__file__).parent / "llm_analyzer"))

from llm_analyzer.models.llm_client import LLMClient
from llm_analyzer.processors.csv_processor import CSVProcessor
from llm_analyzer.config.prompts import PromptsManager

def load_transcript_files(transcripts_path: str) -> List[Path]:
    """Load all CSV transcript files from _data/transcripts folder"""
    transcripts_dir = Path(transcripts_path)
    
    if not transcripts_dir.exists():
        print(f"Error: Transcripts directory {transcripts_path} does not exist")
        sys.exit(1)
    
    csv_files = list(transcripts_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {transcripts_path}")
        sys.exit(1)
    
    return csv_files

def main():
    print("🎙️  OHD LLM Analyzer - CSV Processing Pipeline")
    print("=" * 50)
    
    # Initialize components
    print("Initializing LLM client...")
    llm_client = LLMClient()
    prompts_manager = PromptsManager()
    csv_processor = CSVProcessor(llm_client, prompts_manager)
    
    print(f"Configuration: Processing {prompts_manager.get_batch_size()} rows at a time")
    
    # Load transcript files
    transcript_files = load_transcript_files("_data/transcripts")
    print(f"Found {len(transcript_files)} transcript files to process")
    
    # Phase 1: Extract utterances from all transcripts
    print("\n📝 Phase 1: Extracting significant utterances...")
    all_utterance_groups = []
    
    for i, transcript_file in enumerate(transcript_files, 1):
        print(f"  Processing {transcript_file.name} ({i}/{len(transcript_files)})")
        
        try:
            # Load transcript
            transcript_df = csv_processor.load_transcript_csv(transcript_file)
            print(f"    Loaded {len(transcript_df)} rows")
            
            # Extract utterances (using configured batch size)
            utterances = csv_processor.extract_utterances(transcript_df)
            all_utterance_groups.extend(utterances)
            print(f"    Extracted {len(utterances)} utterance groups")
            
        except Exception as e:
            print(f"    ❌ Error processing {transcript_file.name}: {e}")
            continue
    
    if not all_utterance_groups:
        print("❌ No utterances extracted. Exiting.")
        sys.exit(1)
    
    print(f"✅ Total utterances extracted: {len(all_utterance_groups)}")
    
    # Phase 2: Generate thematic terms
    print("\n🏷️  Phase 2: Generating thematic terms...")
    try:
        thematic_terms = csv_processor.generate_thematic_terms(all_utterance_groups)
        print(f"✅ Generated {len(thematic_terms.terms)} thematic terms:")
        for term in thematic_terms.terms[:10]:  # Show first 10
            print(f"    - {term}")
        if len(thematic_terms.terms) > 10:
            print(f"    ... and {len(thematic_terms.terms) - 10} more")
    except Exception as e:
        print(f"❌ Error generating thematic terms: {e}")
        sys.exit(1)
    
    # Phase 3: Create controlled vocabulary (gen-filters.csv)
    print("\n📚 Phase 3: Building controlled vocabulary...")
    try:
        vocabulary = csv_processor.create_controlled_vocabulary(thematic_terms.terms)
        
        # Save gen-filters.csv
        gen_filters_path = Path("_data/gen-filters.csv")
        csv_processor.save_generated_filters(vocabulary, gen_filters_path)
        print(f"✅ Saved controlled vocabulary to {gen_filters_path}")
        print(f"   Created {len(vocabulary.filters)} filter entries")
        
    except Exception as e:
        print(f"❌ Error creating controlled vocabulary: {e}")
        sys.exit(1)
    
    # Phase 4: Tag transcript rows and save with gen-tags column
    print("\n🏷️  Phase 4: Tagging transcript rows...")
    
    for i, transcript_file in enumerate(transcript_files, 1):
        print(f"  Tagging {transcript_file.name} ({i}/{len(transcript_files)})")
        
        try:
            # Load transcript
            transcript_df = csv_processor.load_transcript_csv(transcript_file)
            
            # Generate tags for rows
            transcript_tags = csv_processor.tag_transcript_rows(
                transcript_df, vocabulary, transcript_file.name
            )
            
            # Validate tag quality
            sample_rows = transcript_df['words'].dropna().head(5).tolist()
            sample_tags = [tag.gen_tags for tag in transcript_tags.row_tags[:5]]
            
            validation = csv_processor.validate_tags_quality(sample_rows, sample_tags)
            
            if not validation.is_valid:
                print(f"    ⚠️  Tag quality warning: {validation.feedback}")
                # Could regenerate tags here if needed
            
            # Save transcript with gen-tags column
            output_path = transcript_file.parent / f"{transcript_file.stem}_tagged.csv"
            csv_processor.save_transcript_with_tags(transcript_df, transcript_tags, output_path)
            
            # Count tagged rows
            tagged_count = sum(1 for tag in transcript_tags.row_tags if tag.gen_tags.strip())
            print(f"    ✅ Tagged {tagged_count}/{len(transcript_df)} rows -> {output_path.name}")
            
        except Exception as e:
            print(f"    ❌ Error tagging {transcript_file.name}: {e}")
            continue
    
    print("\n🎉 Analysis complete!")
    print(f"📁 Check _data/gen-filters.csv for controlled vocabulary")
    print(f"📁 Check _data/transcripts/*_tagged.csv for tagged transcripts")

if __name__ == "__main__":
    main()