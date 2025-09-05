#!/usr/bin/env python3
"""
Simplified main pipeline for CSV transcript analysis.
Processes CSV files in small, manageable steps for robust analysis.
"""
import sys
import os
from pathlib import Path
from simple_llm_client import SimpleLLMClient
from simple_csv_processor import SimpleCSVProcessor


def find_transcript_files(transcripts_dir: str) -> list:
    """Find all CSV files in the transcripts directory"""
    transcripts_path = Path(transcripts_dir)
    
    if not transcripts_path.exists():
        print(f"❌ Transcripts directory '{transcripts_dir}' does not exist")
        return []
    
    csv_files = list(transcripts_path.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in '{transcripts_dir}'")
        return []
    
    print(f"📁 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f.name}")
    
    return csv_files


def process_single_csv(csv_path: Path, processor: SimpleCSVProcessor, output_dir: Path):
    """Process a single CSV file through the complete pipeline"""
    print(f"\n🔄 Processing {csv_path.name}")
    print("=" * 60)
    
    try:
        # Step 1: Load CSV
        df = processor.load_csv(str(csv_path))
        
        # Step 2: Extract meaningful content
        print("\n📝 Step 1: Extracting meaningful content...")
        meaningful_rows = processor.extract_meaningful_content(df)
        
        if not meaningful_rows:
            print("⚠️  No meaningful content found, skipping this file")
            return
        
        # Step 3: Generate vocabulary from meaningful content
        print("\n🏷️  Step 2: Generating vocabulary...")
        vocab_terms = processor.generate_vocabulary_from_content(meaningful_rows)
        
        if not vocab_terms:
            print("⚠️  No vocabulary terms generated, skipping this file")
            return
        
        # Step 4: Create descriptions for vocabulary terms
        print("\n📚 Step 3: Creating vocabulary descriptions...")
        vocab_descriptions = processor.create_vocabulary_descriptions(vocab_terms)
        
        # Step 5: Tag individual rows
        print(f"\n🏷️  Step 4: Tagging {len(df)} individual rows...")
        row_tags = processor.tag_individual_rows(df, vocab_terms, vocab_descriptions)
        
        # Step 6: Save results
        print("\n💾 Step 5: Saving results...")
        
        # Save tagged CSV
        tagged_csv_path = output_dir / f"{csv_path.stem}_tagged.csv"
        processor.save_tagged_csv(df, row_tags, str(tagged_csv_path))
        
        # Save vocabulary for this file
        vocab_csv_path = output_dir / f"{csv_path.stem}_vocabulary.csv"
        processor.save_vocabulary_filters(vocab_terms, vocab_descriptions, str(vocab_csv_path))
        
        print(f"✅ Successfully processed {csv_path.name}")
        print(f"   📄 Tagged CSV: {tagged_csv_path.name}")
        print(f"   📚 Vocabulary: {vocab_csv_path.name}")
        
    except Exception as e:
        print(f"❌ Error processing {csv_path.name}: {e}")
        import traceback
        traceback.print_exc()


def combine_vocabularies(output_dir: Path, csv_files: list):
    """Combine individual vocabulary files into a master gen-filters.csv"""
    print(f"\n📚 Combining vocabularies into master gen-filters.csv...")
    
    all_terms = {}
    
    # Collect all vocabulary terms from individual files
    for csv_file in csv_files:
        vocab_file = output_dir / f"{csv_file.stem}_vocabulary.csv"
        if vocab_file.exists():
            try:
                import pandas as pd
                vocab_df = pd.read_csv(vocab_file)
                for _, row in vocab_df.iterrows():
                    term = row['tag']
                    description = row['description']
                    if term not in all_terms:
                        all_terms[term] = description
            except Exception as e:
                print(f"Warning: Error reading {vocab_file.name}: {e}")
    
    # Save master vocabulary
    if all_terms:
        master_vocab_path = Path("_data/gen-filters.csv")
        master_vocab_path.parent.mkdir(exist_ok=True)
        
        with open(master_vocab_path, 'w', newline='', encoding='utf-8') as f:
            f.write("tag,description\n")
            for term, description in sorted(all_terms.items()):
                f.write(f'"{term}","{description}"\n')
        
        print(f"✅ Created master vocabulary: {master_vocab_path}")
        print(f"   📊 {len(all_terms)} unique terms")
    else:
        print("⚠️  No vocabulary terms found to combine")


def main():
    print("🎙️  Simple OHD LLM Analyzer")
    print("=" * 50)
    print("Processing CSV transcripts with small context windows for robust analysis\n")
    
    # Step 1: Initialize LLM client
    print("🤖 Initializing LLM client...")
    llm_client = SimpleLLMClient()
    
    if not llm_client.test_connection():
        print("❌ LLM client connection failed. Please check your configuration.")
        print("   - For Ollama: Make sure Ollama is running (ollama serve)")
        print("   - For API keys: Check your .env file")
        sys.exit(1)
    
    print(f"✅ Connected to {llm_client.provider} ({llm_client.model})")
    
    # Step 2: Find transcript files
    transcript_files = find_transcript_files("_data/transcripts")
    if not transcript_files:
        sys.exit(1)
    
    # Step 3: Create output directory
    output_dir = Path("_data/transcripts")
    output_dir.mkdir(exist_ok=True)
    
    # Step 4: Initialize processor
    processor = SimpleCSVProcessor(llm_client)
    
    # Step 5: Process each CSV file
    successful_files = []
    for csv_file in transcript_files:
        try:
            process_single_csv(csv_file, processor, output_dir)
            successful_files.append(csv_file)
        except KeyboardInterrupt:
            print("\n🛑 Processing interrupted by user")
            break
        except Exception as e:
            print(f"❌ Fatal error processing {csv_file.name}: {e}")
            continue
    
    # Step 6: Create master vocabulary
    if successful_files:
        combine_vocabularies(output_dir, successful_files)
    
    # Summary
    print(f"\n🎉 Processing complete!")
    print(f"✅ Successfully processed {len(successful_files)} out of {len(transcript_files)} files")
    
    if successful_files:
        print(f"\n📁 Output files:")
        print(f"   📄 Tagged CSVs: *_tagged.csv in _data/transcripts/")
        print(f"   📚 Individual vocabularies: *_vocabulary.csv in _data/transcripts/")
        print(f"   📊 Master vocabulary: _data/gen-filters.csv")
    
    print(f"\n💡 To process a single file: python simple_main.py path/to/file.csv")


def process_single_file(csv_path: str):
    """Process a single CSV file (alternative entry point)"""
    print(f"🎙️  Processing single file: {csv_path}")
    
    # Initialize components
    llm_client = SimpleLLMClient()
    if not llm_client.test_connection():
        print("❌ LLM client connection failed")
        sys.exit(1)
    
    processor = SimpleCSVProcessor(llm_client)
    
    # Process the file
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    output_dir = csv_file.parent
    process_single_csv(csv_file, processor, output_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single file mode
        csv_path = sys.argv[1]
        process_single_file(csv_path)
    else:
        # Batch mode - process all files in _data/transcripts
        main()