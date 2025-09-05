# Simplified OHD LLM Analyzer

A streamlined approach to analyzing oral history transcripts that eliminates Pydantic complexity and uses small context windows for robust processing with any LLM.

## ✅ What's Fixed

### **Eliminated Pydantic Complexity**
- No more JSON schema validation errors
- Simple text-based prompts and responses
- Manual parsing with fallback handling
- Works reliably with smaller models like Ollama

### **Small Context Windows**
- Processes 5 CSV rows at a time for vocabulary generation
- Individual row processing for tagging
- Truncates content to fit within token limits
- Robust error handling at each step

### **Step-by-Step Processing**
1. **Extract Meaningful Content** - Filters out trivial responses
2. **Generate Vocabulary** - Creates terms from small batches
3. **Create Descriptions** - Adds context to vocabulary terms  
4. **Tag Individual Rows** - Applies vocabulary one row at a time
5. **Save Results** - Creates tagged CSV and vocabulary files

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas python-dotenv requests
```

### 2. Configure LLM Provider
Create `.env` file:
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama2
OLLAMA_URL=http://localhost:11434
```

### 3. Test Connection
```bash
python3 simple_llm_client.py
```

### 4. Process CSV Files

**Single file:**
```bash
python3 simple_main.py _data/transcripts/armantrout.csv
```

**All files in directory:**
```bash
python3 simple_main.py
```

### 5. Quick Test
```bash
python3 test_small.py
```

## 📊 Sample Results

From our test with 4 rows of transcript data:

**Generated Vocabulary:**
- `technology`: Tools and platforms impacting writing creation
- `revision`: The iterative process of improving text quality  
- `word processors`: Software tools facilitating writing and editing
- `writing process`: Analyzing stages from idea to final draft
- `computer impact`: How computers altered writing techniques

**Row Tagging:**
```
Row: "Well, I started writing on a typewriter, then moved to computers..."
Tags: technology; word processors; revision; writing process; computer impact
```

## 🔧 Key Features

### **Robust Error Handling**
- Graceful fallbacks when LLM parsing fails
- Continues processing even if individual rows fail
- Multiple LLM provider support with automatic fallbacks

### **Small Context Processing**
- Vocabulary generation: 5 rows at a time
- Row tagging: Individual row processing
- Content truncation to stay within token limits

### **Simple Text Parsing**
- No JSON schema requirements
- Regex-based tag extraction
- Fuzzy matching for vocabulary terms

## 📁 Output Files

- `*_tagged.csv` - Original CSV with `gen-tags` column
- `*_vocabulary.csv` - Vocabulary for each file
- `_data/gen-filters.csv` - Master vocabulary across all files

## 🔄 Processing Flow

```
CSV Input → Meaningful Content → Small Batches → Vocabulary → Descriptions → Row Tagging → Tagged Output
```

Each step processes small chunks of data to ensure reliability with any LLM size or provider.

## 🛠️ Customization

Edit prompts directly in the code files:
- `simple_csv_processor.py` - Contains all prompts
- Modify batch sizes, token limits, and filtering criteria
- Add domain-specific vocabulary guidance

## 🎯 Performance

- **Reliable**: No JSON parsing failures
- **Scalable**: Handles large CSV files in small chunks  
- **Flexible**: Works with Ollama, Claude, OpenAI
- **Robust**: Continues processing despite individual failures

This approach prioritizes reliability and small context windows over complex structured outputs, making it ideal for consistent analysis across different LLM providers and model sizes.