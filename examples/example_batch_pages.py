#!/usr/bin/env python3
"""
Example usage of the batch_pages_convert feature.
This example demonstrates how to use the new batch pages processing capability.
"""
import asyncio
import json
import os
import sys

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def example_batch_pages_convert():
    """Example demonstrating batch pages conversion."""
    
    # Import the function
    from marker_mcp_server.tools import handle_batch_pages_convert
    
    # Example PDF file path
    pdf_file = "/Users/moatasimfarooque/Downloads/marker-1.7.3/uploads/FarooqueMoatasimReportSeptember2023.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ Example PDF not found: {pdf_file}")
        print("Please ensure you have a PDF file to test with.")
        return
    
    # Output directory
    output_dir = "/tmp/marker_batch_pages_example"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔄 Batch Pages Convert Example")
    print("=" * 40)
    print(f"Input PDF: {pdf_file}")
    print(f"Output Directory: {output_dir}")
    print(f"Configuration: Basic (no LLM)")
    
    # Configuration for batch pages processing
    config = {
        "file_path": pdf_file,
        "output_dir": output_dir,
        "pages_per_chunk": 3,  # Small chunks for demonstration
        "combine_output": True,
        "output_format": "markdown",
        "debug": False,
        "use_llm": False,  # Disable LLM to avoid API key requirements
        "config_json": "examples/basic_batch_pages_config.json"
    }
    
    print(f"\nProcessing with {config['pages_per_chunk']} pages per chunk...")
    
    try:
        # Process the document
        result = await handle_batch_pages_convert(config)
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 PROCESSING RESULTS")
        print("=" * 50)
        
        if result.get("success"):
            print("✅ SUCCESS!")
            print(f"📄 Total pages processed: {result.get('total_pages', 'Unknown')}")
            print(f"📦 Total chunks created: {result.get('total_chunks', 'Unknown')}")
            print(f"✅ Successful chunks: {result.get('successful_chunks', 'Unknown')}")
            
            # Show combined output file if available
            if result.get("combined_output_file"):
                combined_file = result["combined_output_file"]
                print(f"📝 Combined output: {combined_file}")
                
                if os.path.exists(combined_file):
                    file_size = os.path.getsize(combined_file)
                    print(f"   File size: {file_size:,} bytes")
                    
                    # Show first few lines of the output
                    try:
                        with open(combined_file, 'r', encoding='utf-8') as f:
                            preview = f.read(500)
                            print(f"\n📋 Preview (first 500 chars):")
                            print("-" * 30)
                            print(preview)
                            if len(preview) == 500:
                                print("...")
                            print("-" * 30)
                    except Exception as e:
                        print(f"   ⚠️ Could not preview file: {e}")
            
            # Show chunk details
            chunk_results = result.get("chunk_results", [])
            if chunk_results:
                print(f"\n📋 Chunk Processing Details:")
                for chunk in chunk_results:
                    status_icon = "✅" if chunk.get("success") else "❌"
                    chunk_num = chunk.get("chunk_num", "?")
                    page_range = chunk.get("page_range", "?")
                    print(f"   {status_icon} Chunk {chunk_num}: Pages {page_range}")
                    
                    if not chunk.get("success") and chunk.get("error"):
                        print(f"      ❌ Error: {chunk.get('error')}")
                    elif chunk.get("output_file"):
                        output_file = chunk.get("output_file")
                        if os.path.exists(output_file):
                            size = os.path.getsize(output_file)
                            print(f"      📄 Output: {os.path.basename(output_file)} ({size:,} bytes)")
        else:
            print("❌ FAILED!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"Message: {result.get('message', 'No message')}")
        
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n📁 Check output directory: {output_dir}")

async def show_feature_info():
    """Display information about the batch pages feature."""
    
    print("🚀 Batch Pages Processing Feature")
    print("=" * 50)
    print("""
This new feature allows you to:

✨ Process large PDF documents efficiently
📦 Split documents into manageable page chunks  
🔧 Process each chunk independently
📝 Combine results into a single output
⚡ Handle memory-intensive documents
🛡️  Provide fault tolerance (if one chunk fails, others continue)
📊 Track detailed progress information

Key Parameters:
• file_path: Path to the PDF file
• pages_per_chunk: Number of pages per chunk (default: 5)
• combine_output: Whether to combine chunks (default: True)
• output_format: markdown, json, or html
• use_llm: Enable LLM for higher quality (requires API keys)

Example configurations are available in:
• examples/basic_batch_pages_config.json (no LLM)
• examples/llm_enhanced_config.json (with LLM)
""")

if __name__ == "__main__":
    print("Marker MCP Server - Batch Pages Processing Example")
    print("=" * 60)
    
    # Show feature information
    asyncio.run(show_feature_info())
    
    # Run the example
    print("\n🎬 Running Example...")
    asyncio.run(example_batch_pages_convert())
    
    print("\n✨ Example completed!")
    print(f"📚 For more information, see: BATCH_PAGES_DOCUMENTATION.md")
