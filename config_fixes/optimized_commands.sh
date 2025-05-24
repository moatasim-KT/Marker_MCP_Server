#!/bin/bash

# Optimized Marker Commands to Reduce LLM Issues and Improve Output Quality
# Run from the marker project directory

echo "=== Optimized Marker Configuration Examples ==="
echo ""

# Basic optimized command without LLM (fastest, most reliable)
echo "1. Basic optimized conversion (no LLM):"
echo "marker_single \"/Users/moatasimfarooque/pdf2markdown_data/Test.pdf\" \\"
echo "  --output_dir output_markdown \\"
echo "  --output_format markdown \\"
echo "  --format_lines \\"
echo "  --extract_images \\"
echo "  --paginate_output \\"
echo "  --lowres_image_dpi 72 \\"
echo "  --highres_image_dpi 150"
echo ""

# Optimized LLM command with reduced concurrency
echo "2. Optimized LLM conversion (reduced payload):"
echo "marker_single \"/Users/moatasimfarooque/pdf2markdown_data/Test.pdf\" \\"
echo "  --output_dir output_markdown \\"
echo "  --output_format markdown \\"
echo "  --use_llm \\"
echo "  --format_lines \\"
echo "  --redo_inline_math \\"
echo "  --extract_images \\"
echo "  --paginate_output \\"
echo "  --max_concurrency 1 \\"
echo "  --max_retries 1 \\"
echo "  --timeout 60 \\"
echo "  --retry_wait_time 10 \\"
echo "  --max_rows_per_batch 15 \\"
echo "  --max_table_rows 40 \\"
echo "  --lowres_image_dpi 72 \\"
echo "  --highres_image_dpi 150 \\"
echo "  --gemini_api_key \$GOOGLE_API_KEY"
echo ""

# Using configuration file
echo "3. Using optimized configuration file:"
echo "marker_single \"/Users/moatasimfarooque/pdf2markdown_data/Test.pdf\" \\"
echo "  --config_json config_fixes/optimized_config.json \\"
echo "  --use_llm \\"
echo "  --gemini_api_key \$GOOGLE_API_KEY"
echo ""

# For very large documents
echo "4. For large documents (chunked processing):"
echo "marker_single \"/Users/moatasimfarooque/pdf2markdown_data/Test.pdf\" \\"
echo "  --output_dir output_markdown \\"
echo "  --output_format markdown \\"
echo "  --use_llm \\"
echo "  --format_lines \\"
echo "  --max_concurrency 1 \\"
echo "  --max_rows_per_batch 10 \\"
echo "  --max_table_rows 25 \\"
echo "  --page_range \"0-4\" \\"
echo "  --timeout 120 \\"
echo "  --gemini_api_key \$GOOGLE_API_KEY"
echo ""

# Using Groq for faster processing
echo "5. Using Groq (potentially faster and cheaper):"
echo "marker_single \"/Users/moatasimfarooque/pdf2markdown_data/Test.pdf\" \\"
echo "  --output_dir output_markdown \\"
echo "  --output_format markdown \\"
echo "  --use_llm \\"
echo "  --llm_service marker.services.groq.GroqService \\"
echo "  --groq_model_name compound-beta \\"
echo "  --groq_api_key \$GROQ_API_KEY \\"
echo "  --format_lines \\"
echo "  --max_concurrency 1 \\"
echo "  --max_rows_per_batch 20 \\"
echo "  --max_groq_tokens 4096"
echo ""

# Table-only processing
echo "6. Table extraction only (reduced scope):"
echo "marker_single \"/Users/moatasimfarooque/pdf2markdown_data/Test.pdf\" \\"
echo "  --converter_cls marker.converters.table.TableConverter \\"
echo "  --use_llm \\"
echo "  --force_layout_block Table \\"
echo "  --output_format json \\"
echo "  --max_rows_per_batch 15 \\"
echo "  --gemini_api_key \$GOOGLE_API_KEY"
echo ""

echo "=== Environment Setup ==="
echo "Make sure to set your API keys:"
echo "export GOOGLE_API_KEY='your_gemini_api_key'"
echo "export GROQ_API_KEY='your_groq_api_key'"
echo ""

echo "=== Troubleshooting Tips ==="
echo "1. If you get rate limit errors:"
echo "   - Reduce --max_concurrency to 1"
echo "   - Increase --retry_wait_time to 15 or 20"
echo "   - Process smaller page ranges with --page_range"
echo ""
echo "2. If you get payload too large errors:"
echo "   - Reduce --max_rows_per_batch to 10 or fewer"
echo "   - Reduce --max_table_rows to 25 or fewer"
echo "   - Lower image DPI settings"
echo ""
echo "3. For better text quality:"
echo "   - Always use --format_lines"
echo "   - Use --redo_inline_math with LLM"
echo "   - Consider --force_ocr for poor quality scans"
echo ""
echo "4. If tables have \\n characters:"
echo "   - The optimized processor cleans these automatically"
echo "   - Use the table-only converter for complex tables"
echo "   - Consider processing in smaller chunks"
echo ""

# Function to run optimized command
run_optimized() {
    local file_path="$1"
    local api_key="$2"
    
    if [ -z "$file_path" ] || [ -z "$api_key" ]; then
        echo "Usage: run_optimized <file_path> <api_key>"
        return 1
    fi
    
    echo "Running optimized conversion for: $file_path"
    
    marker_single "$file_path" \
        --output_dir output_markdown \
        --output_format markdown \
        --use_llm \
        --format_lines \
        --redo_inline_math \
        --extract_images \
        --paginate_output \
        --max_concurrency 1 \
        --max_retries 1 \
        --timeout 60 \
        --retry_wait_time 15 \
        --max_rows_per_batch 15 \
        --max_table_rows 40 \
        --lowres_image_dpi 72 \
        --highres_image_dpi 150 \
        --gemini_api_key "$api_key"
}

echo "=== Quick Run Function ==="
echo "To use the optimized settings quickly:"
echo "source this file, then run:"
echo "run_optimized \"/path/to/your/file.pdf\" \"\$GOOGLE_API_KEY\""