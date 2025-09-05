#!/usr/bin/env python3
"""Single transcript tagging - simplified version for testing"""
import pandas as pd
from pathlib import Path
from tag_transcripts import ThematicTagger
from simple_llm_client import SimpleLLMClient
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 tag_single.py <csv_file>")
        return
    
    csv_file = Path(sys.argv[1])
    if not csv_file.exists():
        print(f"File {csv_file} not found")
        return
    
    print(f"🏷️  Tagging single file: {csv_file.name}")
    
    # Initialize components
    llm_client = SimpleLLMClient()
    if not llm_client.test_connection():
        print("❌ LLM connection failed")
        return
    
    tagger = ThematicTagger(llm_client)
    if not tagger.vocabulary:
        print("❌ No vocabulary loaded")
        return
        
    print(f"✅ Ready with {len(tagger.vocabulary)} vocabulary terms")
    
    # Tag the transcript
    success = tagger.tag_transcript(csv_file)
    
    if success:
        print("✅ Tagging completed successfully")
    else:
        print("❌ Tagging failed")

if __name__ == "__main__":
    main()