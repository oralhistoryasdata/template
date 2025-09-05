#!/usr/bin/env python3
"""
Quick test with minimal data to verify the simplified approach works.
"""
import pandas as pd
from simple_llm_client import SimpleLLMClient
from simple_csv_processor import SimpleCSVProcessor

def create_test_data():
    """Create a small test CSV"""
    test_data = {
        'speaker': ['Devin Becker', 'Rae Armantrout', 'DB', 'RA'],
        'words': [
            'And there we go with that. This could be a good set up.',
            'Mostly poetry. I write poetry and teach poetry.',
            'How has the computer changed your writing process over the years?',
            'Well, I started writing on a typewriter, then moved to computers in the 90s. The revision process became much easier with word processors.'
        ],
        'tags': ['', '', '', ''],
        'timestamp': ['[0:00]', '[01:43]', '[02:15]', '[02:45]']
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('test_sample.csv', index=False)
    return df

def main():
    print("🧪 Testing simplified CSV processor with minimal data...")
    
    # Create test data
    df = create_test_data()
    print(f"Created test CSV with {len(df)} rows")
    
    # Initialize components
    llm_client = SimpleLLMClient()
    if not llm_client.test_connection():
        print("❌ LLM connection failed")
        return
    
    processor = SimpleCSVProcessor(llm_client)
    
    # Test Step 1: Extract meaningful content
    print("\n1️⃣ Testing meaningful content extraction...")
    meaningful_rows = processor.extract_meaningful_content(df, min_words=3)
    print(f"Found {len(meaningful_rows)} meaningful rows")
    
    # Test Step 2: Generate vocabulary (with very small batch)
    print("\n2️⃣ Testing vocabulary generation...")
    try:
        vocab_terms = processor.generate_vocabulary_from_content(meaningful_rows, batch_size=2)
        print(f"Generated vocabulary: {vocab_terms}")
    except Exception as e:
        print(f"Vocabulary generation failed: {e}")
        return
    
    # Test Step 3: Create descriptions
    print("\n3️⃣ Testing vocabulary descriptions...")
    try:
        vocab_descriptions = processor.create_vocabulary_descriptions(vocab_terms)
        print(f"Created {len(vocab_descriptions)} descriptions")
        for term, desc in vocab_descriptions.items():
            print(f"  {term}: {desc}")
    except Exception as e:
        print(f"Description creation failed: {e}")
        vocab_descriptions = {term: f"content about {term}" for term in vocab_terms}
    
    # Test Step 4: Tag a single row
    print("\n4️⃣ Testing row tagging...")
    try:
        # Test with just one meaningful row
        test_row_idx = 3  # The substantive response from RA
        single_row_df = df.iloc[test_row_idx:test_row_idx+1].copy()
        
        row_tags = processor.tag_individual_rows(single_row_df, vocab_terms, vocab_descriptions)
        print(f"Row tags: {row_tags}")
    except Exception as e:
        print(f"Row tagging failed: {e}")
        return
    
    print("\n✅ Basic functionality test completed!")

if __name__ == "__main__":
    main()