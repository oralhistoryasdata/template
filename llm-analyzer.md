# OHD LLM Analyzer

A Python-based tool for analyzing oral history interview transcripts stored in CSV format using Large Language Models (LLMs). This tool extracts thematic content, generates controlled vocabularies, and tags transcript rows with relevant terms.

## Requirements

- **Python 3.7+**: Ensure you have Python 3.7 or higher installed on your system
- **pip**: Python package installer (usually included with Python)
- **LLM Access**: At least one of the following:
  - Local Ollama installation
  - Anthropic API key (for Claude)
  - OpenAI API key

## Features

- **Multi-LLM Support**: Works with Ollama, Claude (Anthropic), and OpenAI
- **CSV Processing**: Processes transcript CSV files with speaker, words, tags, and timestamp columns
- **Utterance Extraction**: Identifies thematically significant content from transcript segments
- **Controlled Vocabulary Generation**: Creates standardized tag vocabularies similar to filters.csv
- **Row-level Tagging**: Tags individual transcript rows with relevant thematic terms
- **Quality Validation**: Evaluates and improves tag quality automatically
- **Output Files**: Generates gen-filters.csv and *_tagged.csv files for Jekyll integration

## Setup Instructions

### 1. Navigate to the LLM Analyzer Directory

From the repository root, change to the `llm_analyzer` directory:

```bash
cd llm_analyzer
```

### 2. Create and Activate Virtual Environment

Create a Python virtual environment to isolate dependencies:

```bash
python3 -m venv venv
```

Activate the virtual environment:

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt, indicating the virtual environment is active.

### 3. Install Dependencies

Install the required Python packages:

```bash
pip3 install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the environment template and configure your settings:

```bash
cp ../.env.example ../.env
```

**Important**: Add the following to your `.gitignore` file (in the repository root) to prevent sensitive information from being committed:

```gitignore
# Environment variables (contains API keys)
.env

# Virtual environment
llm_analyzer/venv/
llm_analyzer/venv

# Python cache
__pycache__/
*.pyc
*.pyo
```

Edit the `.env` file in the repository root with your preferred LLM provider:

**For Ollama (local):**
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama2
OLLAMA_URL=http://localhost:11434
```

**For Claude:**
```env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your_api_key_here
```

**For OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### 5. Prepare Your Data

Ensure your transcript CSV files are in the `_data/transcripts/` folder (one directory up from this folder). The tool expects CSV files with this structure:

```csv
speaker,words,tags,timestamp
Devin Becker,"And there we go with that...",,[0:00]
Rae Armantrout,"Mostly poetry.",,[01:43]
```

**Required columns:**
- `speaker`: Name of the person speaking
- `words`: The spoken words/content
- `tags`: Existing manual tags (can be empty)
- `timestamp`: Time marker (optional, can be empty)

### 6. Run the Analyzer

Execute the main analysis pipeline:

```bash
python3 main.py
```

The tool will:
1. Load CSV transcript files from `../_data/transcripts/`
2. Extract significant utterances from transcript segments (100 rows at a time)
3. Generate thematic terms from utterances
4. Create a controlled vocabulary and save to `../_data/gen-filters.csv`
5. Tag individual transcript rows with relevant terms
6. Save tagged transcripts as `*_tagged.csv` files with new `gen-tags` column

## Output Files

### gen-filters.csv
Contains the generated controlled vocabulary in the same format as your existing `filters.csv`:
```csv
tag,description
between,working between media to advance writing process
paper,using paper in the writing process
revision,revision practices and methods
correspondence,correspondence with other writers
```

### *_tagged.csv files
Contains your original transcript data with an additional `gen-tags` column:
```csv
speaker,words,tags,timestamp,gen-tags
Devin Becker,"And there we go with that...",,[0:00],
Rae Armantrout,"Mostly poetry.",,[01:43],"revision; correspondence"
```

The `gen-tags` column contains semicolon-separated tags that were automatically generated based on the controlled vocabulary.

## Configuration

### Prompts and Settings

You can customize the analysis behavior by editing `config/prompts.yml`:

```yaml
# Configuration settings
config:
  batch_size: 100  # Number of transcript rows to process at once
  vocab_terms_range: "3-6"  # Number of vocabulary terms to generate per batch

# All prompts used by the LLM for different stages
prompts:
  extract_utterances:
    user: |
      Extract {vocab_terms_range} significant thematic utterances...
```

**Key settings:**
- `batch_size`: Controls how many transcript rows are processed together (default: 100)
- `vocab_terms_range`: Range of vocabulary terms to generate per batch (default: "3-6")

**Customizable prompts:**
- `extract_utterances`: How the system identifies significant content
- `generate_thematic_terms`: How thematic terms are created
- `create_controlled_vocabulary`: How the controlled vocabulary is built
- `tag_transcript_rows`: How individual rows are tagged
- `validate_tags_quality`: How tag quality is evaluated

## Deactivating the Virtual Environment

When you're finished working with the tool, deactivate the virtual environment:

```bash
deactivate
```

The `(venv)` indicator will disappear from your terminal prompt.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the virtual environment is activated and dependencies are installed
2. **API Key Errors**: Verify your API keys are correctly set in the `.env` file
3. **Ollama Connection**: Ensure Ollama is running locally if using the ollama provider
4. **Data Format**: Check that your transcript CSV files have the expected structure (speaker, words, tags, timestamp columns)

### Getting Help

- Check that your CSV files are in the `_data/transcripts/` folder
- Verify your CSV files have the required columns: speaker, words, tags, timestamp
- Ensure your chosen LLM provider is properly configured

## Customization

To customize the analysis for your specific content:

1. **Modify Prompts**: Edit `config/prompts.yml` to adjust all LLM prompts and processing settings
2. **Adjust Batch Size**: Change `batch_size` in `prompts.yml` to process more/fewer rows at once
3. **Tune Vocabulary Generation**: Modify `vocab_terms_range` to generate different numbers of terms per batch

## Project Structure

```
llm_analyzer/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── main.py               # Main analysis pipeline
├── config/
│   ├── settings.py       # LLM configuration management
│   ├── prompts.yml       # Prompts and processing settings
│   └── prompts.py        # Prompts manager
├── models/
│   ├── llm_client.py     # LLM communication
│   └── structured_outputs.py # Pydantic models for CSV processing
└── processors/
    └── csv_processor.py  # CSV transcript processing
```