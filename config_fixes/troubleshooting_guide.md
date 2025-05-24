# Marker LLM Issues Troubleshooting Guide

## Common Issues and Solutions

### 1. High Payload Errors and Rate Limiting

**Symptoms:**
- "Request too large" errors
- Rate limit exceeded (429 errors)
- Timeout errors
- Failed API calls

**Root Causes:**
- Large images being sent to LLM
- Long HTML tables in prompts
- Too many concurrent requests
- Default settings optimized for speed, not reliability

**Solutions:**

#### Immediate Fixes:
```bash
# Reduce concurrency and batch sizes
marker_single "your_file.pdf" \
  --use_llm \
  --max_concurrency 1 \
  --max_rows_per_batch 10 \
  --max_table_rows 25 \
  --timeout 120 \
  --retry_wait_time 15 \
  --lowres_image_dpi 72 \
  --highres_image_dpi 150
```

#### For Very Large Documents:
```bash
# Process in smaller chunks
marker_single "your_file.pdf" \
  --use_llm \
  --page_range "0-2" \
  --max_concurrency 1 \
  --max_rows_per_batch 5 \
  --max_table_rows 15
```

#### Alternative LLM Services:
```bash
# Try Groq (often more reliable)
marker_single "your_file.pdf" \
  --use_llm \
  --llm_service marker.services.groq.GroqService \
  --groq_api_key $GROQ_API_KEY \
  --groq_model_name compound-beta \
  --max_groq_tokens 4096
```

### 2. Poor Output Quality with \n Characters

**Symptoms:**
- Raw `\n`, `\t`, `\r` characters in output
- Broken table formatting
- Escaped characters in text
- Poor text flow

**Root Causes:**
- OCR artifacts not cleaned
- HTML parsing issues
- LLM responses not properly processed
- Text processing pipeline issues

**Solutions:**

#### Use Optimized Processing:
```bash
marker_single "your_file.pdf" \
  --format_lines \
  --redo_inline_math \
  --use_llm \
  --extract_images \
  --paginate_output
```

#### For Poor Quality Scans:
```bash
marker_single "your_file.pdf" \
  --force_ocr \
  --format_lines \
  --strip_existing_ocr
```

#### Table-Specific Issues:
```bash
# Use table-only converter for complex tables
marker_single "your_file.pdf" \
  --converter_cls marker.converters.table.TableConverter \
  --use_llm \
  --output_format json \
  --max_rows_per_batch 10
```

### 3. Chunking Problems

**Current Issues:**
- Default `max_rows_per_batch: 60` is too large
- No cell count limits
- No payload size validation
- Poor error recovery

**Optimized Settings:**
```bash
# Smaller, more manageable chunks
marker_single "your_file.pdf" \
  --use_llm \
  --max_rows_per_batch 15 \
  --max_table_rows 40 \
  --table_rec_batch_size 1 \
  --layout_batch_size 1
```

## Implementation of Fixes

### 1. Apply the Optimized Configuration

Copy the optimized config and use it:
```bash
marker_single "your_file.pdf" \
  --config_json config_fixes/optimized_config.json \
  --use_llm \
  --gemini_api_key $GOOGLE_API_KEY
```

### 2. Environment Variables for Better Defaults

Add to your `~/.zshrc`:
```bash
# Marker optimized defaults
export MARKER_MAX_CONCURRENCY=1
export MARKER_MAX_RETRIES=1
export MARKER_TIMEOUT=60
export MARKER_MAX_ROWS_PER_BATCH=15
export MARKER_MAX_TABLE_ROWS=40
export GOOGLE_API_KEY="your_api_key_here"
export GROQ_API_KEY="your_groq_key_here"
```

### 3. Processing Strategy by Document Type

#### Small Documents (< 10 pages):
```bash
marker_single "file.pdf" \
  --use_llm \
  --format_lines \
  --redo_inline_math \
  --max_concurrency 1
```

#### Medium Documents (10-50 pages):
```bash
marker_single "file.pdf" \
  --use_llm \
  --format_lines \
  --max_concurrency 1 \
  --max_rows_per_batch 15 \
  --timeout 120
```

#### Large Documents (50+ pages):
```bash
# Process in chunks
for i in {0..9}; do
  start=$((i * 10))
  end=$((start + 9))
  marker_single "file.pdf" \
    --use_llm \
    --page_range "${start}-${end}" \
    --output_dir "output_${start}_${end}" \
    --max_concurrency 1 \
    --max_rows_per_batch 10
done
```

#### Table-Heavy Documents:
```bash
marker_single "file.pdf" \
  --converter_cls marker.converters.table.TableConverter \
  --use_llm \
  --max_rows_per_batch 8 \
  --max_table_rows 20 \
  --output_format json
```

## Performance Optimization

### 1. Without LLM (Fastest):
```bash
marker_single "file.pdf" \
  --format_lines \
  --extract_images \
  --lowres_image_dpi 96 \
  --highres_image_dpi 192
```

### 2. With LLM (Balanced):
```bash
marker_single "file.pdf" \
  --use_llm \
  --format_lines \
  --redo_inline_math \
  --max_concurrency 1 \
  --max_rows_per_batch 15 \
  --lowres_image_dpi 72
```

### 3. High Quality (Slower):
```bash
marker_single "file.pdf" \
  --use_llm \
  --format_lines \
  --redo_inline_math \
  --force_ocr \
  --max_concurrency 1 \
  --max_rows_per_batch 10 \
  --max_table_rows 25 \
  --timeout 180
```

## Monitoring and Debugging

### 1. Enable Debug Mode:
```bash
marker_single "file.pdf" \
  --debug \
  --debug_data_folder debug_output \
  --debug_layout_images \
  --debug_json
```

### 2. Check Token Usage:
The debug output will show:
- `llm_tokens_used`: Tokens consumed per request
- `llm_request_count`: Number of API calls
- `llm_error_count`: Failed requests

### 3. Log Analysis:
```bash
# Check for common issues
grep -i "rate limit\|timeout\|payload\|error" marker_log.txt
```

## Advanced Troubleshooting

### 1. Custom LLM Service Settings:
```bash
# For Gemini with stricter limits
marker_single "file.pdf" \
  --use_llm \
  --gemini_model_name "gemini-2.0-flash" \
  --timeout 60 \
  --max_retries 1 \
  --retry_wait_time 20
```

### 2. Alternative Processing Approaches:
```bash
# OCR-only first, then post-process
marker_single "file.pdf" \
  --converter_cls marker.converters.ocr.OCRConverter \
  --format_lines \
  --keep_chars

# Then process the text output with LLM separately
```

### 3. Batch Processing Script:
```bash
#!/bin/zsh
# Process multiple files with error recovery
for file in *.pdf; do
  echo "Processing: $file"
  marker_single "$file" \
    --use_llm \
    --max_concurrency 1 \
    --max_rows_per_batch 15 \
    --timeout 120 \
    --output_dir "output_$(basename "$file" .pdf)" \
    || echo "Failed: $file" >> failed_files.txt
  sleep 30  # Cool-down period
done
```

## Quick Fix Commands

### Immediate Relief:
```bash
# Your current command with fixes
marker_single "/Users/moatasimfarooque/pdf2markdown_data/Test.pdf" \
  --output_dir output_markdown \
  --output_format markdown \
  --format_lines \
  --extract_images \
  --use_llm \
  --max_concurrency 1 \
  --max_rows_per_batch 10 \
  --max_table_rows 25 \
  --timeout 120 \
  --retry_wait_time 15 \
  --lowres_image_dpi 72 \
  --gemini_api_key $GOOGLE_API_KEY
```

### For Better Text Quality:
```bash
marker_single "/Users/moatasimfarooque/pdf2markdown_data/Test.pdf" \
  --output_dir output_markdown \
  --output_format markdown \
  --format_lines \
  --redo_inline_math \
  --use_llm \
  --extract_images \
  --paginate_output \
  --max_concurrency 1 \
  --max_rows_per_batch 15 \
  --gemini_api_key $GOOGLE_API_KEY
```

## Expected Improvements

After applying these fixes:
- ✅ Reduced rate limiting errors
- ✅ Smaller, more manageable API payloads
- ✅ Better text quality with cleaned `\n` characters
- ✅ More reliable table processing
- ✅ Improved error recovery
- ✅ Better chunking strategy
- ✅ More predictable processing times
