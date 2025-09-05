#!/usr/bin/env python3
"""
Simplified CSV processor that works with small context windows.
Step-by-step processing for robust analysis.
"""
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from simple_llm_client import SimpleLLMClient


class SimpleCSVProcessor:
    def __init__(self, llm_client: SimpleLLMClient = None):
        self.llm = llm_client or SimpleLLMClient()
        self.vocabulary = set()
        self.vocabulary_descriptions = {}
        
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV file with error handling"""
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from {csv_path}")
            
            # Ensure required columns exist
            required_cols = ['speaker', 'words']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols}. Adding empty columns.")
                for col in missing_cols:
                    df[col] = ''
            
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV {csv_path}: {e}")
    
    def extract_meaningful_content(self, df: pd.DataFrame, min_words: int = 5) -> List[Dict]:
        """Extract rows with meaningful content for analysis"""
        meaningful_rows = []
        
        for idx, row in df.iterrows():
            words = str(row.get('words', '')).strip()
            speaker = str(row.get('speaker', '')).strip()
            
            # Skip empty, very short, or purely structural content
            if (len(words) > min_words and 
                len(words.split()) >= 3 and  # At least 3 words
                not words.startswith('[') and  # Skip timestamps/markers
                words not in ['Good.', 'OK.', 'Right.', 'Yes.', 'No.']):  # Skip minimal responses
                
                meaningful_rows.append({
                    'index': idx,
                    'speaker': speaker,
                    'words': words[:500],  # Limit length for context window
                    'word_count': len(words.split())
                })
        
        print(f"Found {len(meaningful_rows)} meaningful rows out of {len(df)}")
        return meaningful_rows
    
    def generate_vocabulary_from_content(self, meaningful_rows: List[Dict], batch_size: int = 5) -> Set[str]:
        """Generate vocabulary by analyzing small batches of meaningful content"""
        all_vocab_terms = set()
        
        print(f"Generating vocabulary from {len(meaningful_rows)} meaningful rows...")
        
        # Process in small batches to stay within context limits
        for i in range(0, len(meaningful_rows), batch_size):
            batch = meaningful_rows[i:i+batch_size]
            
            # Create focused context for this batch
            batch_text = "\n".join([
                f"{row['speaker']}: {row['words']}"
                for row in batch
            ])
            
            prompt = f"""Analyze this oral history transcript segment and identify 3-5 key thematic terms that could be used as tags.

Focus on themes about:
- Writing and creative processes
- Technology's impact on creative work  
- Professional development in the arts
- Changes in practice over time

Transcript segment:
{batch_text}

List only the most important thematic terms, one per line, like this:
- revision
- correspondence  
- technology
- paper
- early

Keep terms short (1-3 words) and suitable for tagging."""

            try:
                response = self.llm.generate_text(prompt, max_tokens=200)
                batch_terms = self.parse_vocabulary_terms(response)
                all_vocab_terms.update(batch_terms)
                print(f"  Batch {i//batch_size + 1}: extracted {len(batch_terms)} terms")
                
            except Exception as e:
                print(f"  Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"Generated {len(all_vocab_terms)} unique vocabulary terms")
        return all_vocab_terms
    
    def parse_vocabulary_terms(self, llm_response: str) -> List[str]:
        """Parse vocabulary terms from LLM response"""
        terms = []
        
        # Look for terms in various formats
        lines = llm_response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Handle bullet points, numbers, dashes
            if re.match(r'^[-*•]\s*(.+)', line):
                term = re.match(r'^[-*•]\s*(.+)', line).group(1).strip()
            elif re.match(r'^\d+\.\s*(.+)', line):
                term = re.match(r'^\d+\.\s*(.+)', line).group(1).strip()
            elif line and not line.startswith(('Here', 'The', 'These', 'Based')):
                term = line
            else:
                continue
            
            # Clean and validate term
            term = term.lower().strip('.,;:"\'')
            if term and len(term.split()) <= 3 and len(term) > 2:
                terms.append(term)
        
        return terms[:6]  # Limit to prevent overwhelming the model
    
    def create_vocabulary_descriptions(self, vocab_terms: Set[str]) -> Dict[str, str]:
        """Create descriptions for vocabulary terms in small batches"""
        descriptions = {}
        
        # Process terms in very small batches to create focused descriptions
        terms_list = list(vocab_terms)
        batch_size = 3
        
        print(f"Creating descriptions for {len(terms_list)} vocabulary terms...")
        
        for i in range(0, len(terms_list), batch_size):
            batch_terms = terms_list[i:i+batch_size]
            
            prompt = f"""Create brief, clear descriptions for these thematic terms used in oral history analysis about writing and technology.

Terms to define:
{chr(10).join([f'- {term}' for term in batch_terms])}

For each term, provide a short description (5-10 words) explaining how it relates to writing practices or creative work.

Format your response like this:
term1: description of the term
term2: description of the term
term3: description of the term"""

            try:
                response = self.llm.generate_text(prompt, max_tokens=300)
                batch_descriptions = self.parse_term_descriptions(response)
                descriptions.update(batch_descriptions)
                print(f"  Created descriptions for batch {i//batch_size + 1}")
                
            except Exception as e:
                print(f"  Error creating descriptions for batch {i//batch_size + 1}: {e}")
                # Create fallback descriptions
                for term in batch_terms:
                    descriptions[term] = f"thematic content related to {term}"
        
        return descriptions
    
    def parse_term_descriptions(self, llm_response: str) -> Dict[str, str]:
        """Parse term descriptions from LLM response"""
        descriptions = {}
        
        lines = llm_response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for "term: description" pattern
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    term = parts[0].strip().lower().strip('- ')
                    description = parts[1].strip()
                    if term and description:
                        descriptions[term] = description
        
        return descriptions
    
    def tag_individual_rows(self, df: pd.DataFrame, vocab_terms: Set[str], vocab_descriptions: Dict[str, str]) -> Dict[int, str]:
        """Tag individual CSV rows using the vocabulary"""
        row_tags = {}
        
        # Create vocabulary context (limit size for context window)
        vocab_list = list(vocab_terms)[:15]  # Limit vocabulary size for each request
        vocab_context = "\n".join([
            f"- {term}: {vocab_descriptions.get(term, 'thematic term')}"
            for term in vocab_list
        ])
        
        print(f"Tagging individual rows with vocabulary of {len(vocab_list)} terms...")
        
        # Process each meaningful row individually to ensure focused analysis
        for idx, row in df.iterrows():
            words = str(row.get('words', '')).strip()
            speaker = str(row.get('speaker', '')).strip()
            
            # Skip rows without meaningful content
            if len(words) < 10 or len(words.split()) < 3:
                row_tags[idx] = ""
                continue
            
            # Truncate very long content to fit context window
            content = words[:400]
            
            prompt = f"""Tag this oral history transcript row with relevant terms from the vocabulary below.
Only use terms that clearly apply to the content. Use semicolon separation if multiple terms apply.

Available terms:
{vocab_context}

Row content:
Speaker: {speaker}
Words: {content}

Response format: Just list the applicable terms separated by semicolons, or leave blank if no terms apply.
Example: revision; paper
Example: technology
Example: (blank if no terms apply)

Tags:"""

            try:
                response = self.llm.generate_text(prompt, max_tokens=100)
                tags = self.parse_row_tags(response, vocab_terms)
                row_tags[idx] = tags
                
                if tags:
                    print(f"  Row {idx}: '{words[:50]}...' -> {tags}")
                    
            except Exception as e:
                print(f"  Error tagging row {idx}: {e}")
                row_tags[idx] = ""
                continue
        
        tagged_count = sum(1 for tags in row_tags.values() if tags.strip())
        print(f"Tagged {tagged_count} out of {len(df)} rows")
        
        return row_tags
    
    def parse_row_tags(self, llm_response: str, valid_vocab: Set[str]) -> str:
        """Parse and validate tags from LLM response"""
        response = llm_response.strip()
        
        # Clean up the response
        response = response.replace('Tags:', '').strip()
        response = re.sub(r'^(Response:|Answer:|Tags are:|The tags are:)', '', response).strip()
        
        if not response or response.lower() in ['none', 'blank', 'no tags', '(blank)']:
            return ""
        
        # Split on semicolons or commas
        potential_tags = re.split(r'[;,]', response)
        
        valid_tags = []
        for tag in potential_tags:
            tag = tag.strip().lower()
            tag = re.sub(r'^[-*•]\s*', '', tag)  # Remove bullet points
            
            # Check if tag is in our vocabulary (fuzzy match)
            if tag in valid_vocab:
                valid_tags.append(tag)
            else:
                # Check for partial matches
                for vocab_term in valid_vocab:
                    if tag in vocab_term or vocab_term in tag:
                        valid_tags.append(vocab_term)
                        break
        
        return "; ".join(valid_tags) if valid_tags else ""
    
    def save_tagged_csv(self, df: pd.DataFrame, row_tags: Dict[int, str], output_path: str):
        """Save CSV with gen-tags column"""
        # Add gen-tags column
        df['gen-tags'] = df.index.map(lambda x: row_tags.get(x, ""))
        
        # Save to file
        df.to_csv(output_path, index=False)
        
        tagged_count = sum(1 for tags in row_tags.values() if tags.strip())
        print(f"Saved tagged CSV to {output_path} ({tagged_count} rows tagged)")
    
    def save_vocabulary_filters(self, vocab_terms: Set[str], vocab_descriptions: Dict[str, str], output_path: str):
        """Save vocabulary as gen-filters.csv"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            f.write("tag,description\n")
            for term in sorted(vocab_terms):
                description = vocab_descriptions.get(term, f"thematic content related to {term}")
                f.write(f'"{term}","{description}"\n')
        
        print(f"Saved vocabulary filters to {output_path} ({len(vocab_terms)} terms)")