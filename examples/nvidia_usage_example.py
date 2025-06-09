#!/usr/bin/env python3
"""
Example script demonstrating how to use the NVIDIA service for PDF conversion.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marker.converters.pdf import PdfConverter
from marker.services.nvidia import NvidiaService


def convert_pdf_with_nvidia(
    pdf_path: str,
    output_path: str = None,
    nvidia_api_key: str = None,
):
    """
    Convert a PDF using the NVIDIA service.

    Args:
        pdf_path: Path to the PDF file
        output_path: Output path for the converted file (optional)
        nvidia_api_key: NVIDIA API key (optional, can be set via environment variable)

    Returns:
        Converted document
    """
    
    # Get API key from parameter or environment variable
    api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA API key is required. Set NVIDIA_API_KEY environment variable or pass it as parameter.")
        return None

    # Create NVIDIA service configuration
    nvidia_config = {
        "nvidia_api_key": api_key,
        "nvidia_model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
        "timeout": 60,
        "max_retries": 3,
        "retry_wait_time": 5,
    }

    # Create the converter with NVIDIA service
    converter = PdfConverter()
    converter.llm_service = NvidiaService(nvidia_config)

    # Convert the document
    print(f"Converting {pdf_path} using NVIDIA service...")
    document = converter(pdf_path)

    # Generate output
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(document.render())
        print(f"Converted document saved to {output_path}")
    else:
        print("Conversion completed. Use output_path parameter to save the result.")

    return document


def main():
    """Main example function."""
    import argparse

    parser = argparse.ArgumentParser(description="NVIDIA service PDF conversion example")
    parser.add_argument("pdf_path", help="Path to the PDF file to convert")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--api-key", help="NVIDIA API key (or set NVIDIA_API_KEY env var)")

    args = parser.parse_args()

    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found")
        return 1

    # Set up output path
    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_path = f"{base_name}_nvidia.md"

    try:
        # Convert the document
        document = convert_pdf_with_nvidia(
            pdf_path=args.pdf_path,
            output_path=output_path,
            nvidia_api_key=args.api_key,
        )

        if document:
            print("\nConversion completed successfully!")
            print("Features used:")
            print("  ✓ NVIDIA Llama-3.1-Nemotron-Nano-VL-8B-V1 model")
            print("  ✓ Vision-language processing")
            print("  ✓ Advanced layout understanding")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
