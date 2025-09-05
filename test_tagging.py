#!/usr/bin/env python3
"""Test script for thematic tagging pipeline"""
import pandas as pd
from pathlib import Path
from tag_transcripts import ThematicTagger
from simple_llm_client import SimpleLLMClient

def main():
    print("🧪 Testing Thematic Tagging Pipeline")
    print("=" * 40)
    
    # Initialize LLM client
    llm_client = SimpleLLMClient()
    if not llm_client.test_connection():
        print("❌ LLM connection failed")
        return
    print(f"✅ LLM connected: {llm_client.provider}")
    
    # Initialize tagger  
    tagger = ThematicTagger(llm_client)
    print(f"✅ Loaded {len(tagger.vocabulary)} vocabulary terms")
    
    # Load a small sample of transcript data
    csv_path = Path("_data/transcripts/armantrout.csv")
    if not csv_path.exists():
        print(f"❌ {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Take just the first 20 rows for testing
    test_df = df.head(20).copy()
    print(f"📊 Testing with {len(test_df)} rows")
    
    # Test section creation
    sections = tagger.create_thematic_sections(test_df)
    print(f"📝 Created {len(sections)} sections")
    
    if sections:
        # Test analysis of first section only
        first_section = sections[0]
        print(f"\n🔍 Testing analysis of first section (rows {first_section['start_row']}-{first_section['end_row']})")
        print("Preview:")
        print(first_section['section_text'][:200] + "...")
        
        analysis = tagger.analyze_section_themes(first_section)
        print(f"\n✅ Analysis result:")
        print(f"  Themes: {analysis['themes']}")
        print(f"  Confidence: {analysis['confidence']}")
        print(f"  Reasoning: {analysis['reasoning']}")


if __name__ == "__main__":
    main()