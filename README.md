## Financial QA: RAG vs Fine-Tuning

**Group 122 - Advanced RAG Technique: Chunk Merging & Adaptive Retrieval**

### Quick Start
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Add documents**: Place financial documents in `financial_data/` directory
3. **Test processing**: `python test_enhanced_processing.py`
4. **Process documents**: `python -c "from financial_qa.enhanced_data_processing import main; main()"`
5. **Run UI**: `streamlit run app_streamlit.py`

### Enhanced Data Collection & Preprocessing

#### 📄 **Multi-Format Document Support**
- **PDF**: Text extraction + OCR fallback for image-based pages
- **Excel**: Spreadsheet parsing with sheet identification
- **HTML**: Clean text extraction with noise removal
- **Word**: Document text extraction

#### 🧹 **Advanced Text Cleaning**
- Remove headers, footers, page numbers
- Clean excessive whitespace and special characters
- Fix financial formatting (commas, dollar signs)
- Noise reduction and standardization

#### 📊 **Logical Section Segmentation**
- **Income Statement**: Revenue, profit, earnings data
- **Balance Sheet**: Assets, liabilities, equity
- **Cash Flow**: Operating, investing, financing activities
- **Operational**: R&D, administrative, marketing expenses

#### ❓ **Automatic Q/A Pair Generation**
- **50+ Template Questions**: Revenue, profit, assets, expenses by year
- **Smart Answer Extraction**: Pattern-based financial data extraction
- **Category Classification**: Revenue, balance sheet, operational, growth
- **Year-Specific Queries**: Automatic year detection and matching

### Data Source
- **Input**: Multiple document types in `financial_data/` directory
- **Output**: 
  - `processed_financial_data.json` - Structured document data
  - `qa_pairs.jsonl` - Training data for fine-tuning
  - `merged_financial_reports.txt` - Consolidated text for RAG
- **Processing**: Multi-format → Clean text → Logical sections → Q/A pairs → Chunks

### Features
- **RAG Mode**: Hybrid retrieval (FAISS + BM25) with adaptive chunk selection
- **Fine-Tuned Mode**: LoRA fine-tuned FLAN-T5-large
- **Advanced RAG**: Chunk merging & adaptive retrieval based on query complexity
- **Guardrails**: Input validation and hallucination detection
- **OCR Support**: Handles image-based PDFs and scanned documents

### Structure
```
financial_qa/
├── enhanced_data_processing.py  # Multi-format processing + OCR
├── data_processing.py           # Chunking and integration
├── retrieval.py                 # Hybrid retrieval + adaptive
├── generate.py                  # T5 generation
└── finetune.py                 # LoRA fine-tuning
```

### Document Processing Pipeline
1. **Discovery**: Scan for PDF, Excel, HTML, Word files
2. **Extraction**: Text extraction with format-specific parsers
3. **OCR Fallback**: Image-based PDF handling with Tesseract
4. **Cleaning**: Noise removal and formatting standardization
5. **Segmentation**: Logical financial section identification
6. **Q/A Generation**: Template-based question-answer creation
7. **Chunking**: Intelligent text splitting for retrieval

### Fine-tuning
1. **Automatic Q/A Generation**: System creates 50+ training pairs
2. **Manual Enhancement**: Edit `data/qa_pairs.jsonl` for custom questions
3. **Run Training**: `python -c "from financial_qa.finetune import finetune_lora; finetune_lora()"`
4. **Test**: Restart app and select "Fine-Tuned" mode

### Advanced RAG (Group 122)
- **Query Complexity Analysis**: Determines optimal chunk size preference
- **Dynamic Chunk Merging**: Combines adjacent small chunks when beneficial
- **Adaptive Parameters**: Adjusts retrieval depth and fusion weights per query
- **Multi-Format Context**: Leverages structured section information

### Guardrails
- Input relevance scoring (triggers "Data not in scope" for irrelevant queries)
- Output validation to detect potential hallucinations
- Financial data verification against source documents

### System Requirements
- **OCR**: Tesseract for image-based document processing
- **Memory**: 8GB+ RAM recommended for large document processing
- **Storage**: 2GB+ free space for processed data and models
- **Python**: 3.8+ with virtual environment support

### Troubleshooting
- **OCR Issues**: Install Tesseract: `sudo apt-get install tesseract-ocr`
- **Memory Issues**: Reduce chunk sizes in `config.py`
- **Format Issues**: Check document encoding and structure
- **Q/A Quality**: Review and edit generated pairs in `qa_pairs.jsonl`
