#!/usr/bin/env python3
"""
Example script demonstrating the enhanced PDF conversion with improved
heading detection and caption handling.
"""

import os

# Import our enhanced converter
from marker.converters.enhanced_pdf import EnhancedPdfConfig, EnhancedPdfConverter
from marker.schema import BlockTypes
from marker.services.claude import ClaudeService
from marker.services.gemini import GoogleGeminiService
from marker.services.nvidia import NvidiaService
from marker.services.openai import OpenAIService


def convert_pdf_with_enhancements(
    pdf_path: str,
    output_path: str = None,
    use_llm: bool = True,
    llm_service: str = "gemini",
    config_overrides: dict = None,
):
    """
    Convert a PDF using the enhanced converter with improved heading and caption detection.

    Args:
        pdf_path: Path to the PDF file
        output_path: Output path for the converted file (optional)
        use_llm: Whether to use LLM-based enhancements
        llm_service: Which LLM service to use ("openai", "claude", "gemini", "nvidia")
        config_overrides: Dictionary of configuration overrides

    Returns:
        Converted document
    """

    # Create configuration
    config = EnhancedPdfConfig()
    config.use_llm = use_llm

    # Apply any configuration overrides
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    # Create the enhanced converter
    converter = EnhancedPdfConverter(config)

    # Set up LLM service if requested
    if use_llm:
        if llm_service == "openai":
            converter.llm_service = OpenAIService(config)
        elif llm_service == "claude":
            converter.llm_service = ClaudeService(config)
        elif llm_service == "gemini":
            converter.llm_service = GoogleGeminiService(config)
        elif llm_service == "nvidia":
            converter.llm_service = NvidiaService(config)
        else:
            print(f"Warning: Unknown LLM service '{llm_service}', using Gemini")
            converter.llm_service = GoogleGeminiService(config)

    # Convert the document
    print(f"Converting {pdf_path} with enhanced processing...")
    document = converter(pdf_path)

    # Generate output
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(document.render())
        print(f"Converted document saved to {output_path}")

    return document


def analyze_conversion_improvements(document):
    """
    Analyze the improvements made by the enhanced conversion.

    Args:
        document: The converted document

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "total_pages": len(document.pages),
        "headings": [],
        "captions": [],
        "enhanced_blocks": [],
        "heading_levels": {},
        "caption_visual_pairs": 0,
    }

    for page in document.pages:
        # Count headings and their levels
        for block in page.contained_blocks(document, (BlockTypes.SectionHeader,)):
            heading_info = {
                "text": block.raw_text(document).strip(),
                "level": getattr(block, "heading_level", None),
                "page": page.page_id,
                "enhanced": hasattr(block, "metadata")
                and block.metadata
                and getattr(block.metadata, "llm_layout_refinement", False),
            }
            analysis["headings"].append(heading_info)

            # Count by level
            level = heading_info["level"] or "unknown"
            analysis["heading_levels"][level] = (
                analysis["heading_levels"].get(level, 0) + 1
            )

        # Count captions
        for block in page.contained_blocks(document, (BlockTypes.Caption,)):
            caption_info = {
                "text": block.raw_text(document).strip()[:100] + "...",
                "page": page.page_id,
                "enhanced": hasattr(block, "metadata")
                and block.metadata
                and getattr(block.metadata, "llm_layout_refinement", False),
            }
            analysis["captions"].append(caption_info)

        # Count blocks that were enhanced by our processors
        for block in page.children:
            if (
                hasattr(block, "metadata")
                and block.metadata
                and getattr(block.metadata, "llm_layout_refinement", False)
            ):
                analysis["enhanced_blocks"].append(
                    {
                        "type": str(block.block_type),
                        "original_type": getattr(
                            block.metadata, "original_block_type", "unknown"
                        ),
                        "reasoning": getattr(
                            block.metadata, "llm_refinement_reasoning", ""
                        ),
                        "page": page.page_id,
                    }
                )

    return analysis


def print_analysis_report(analysis):
    """Print a detailed analysis report."""
    print("\n" + "=" * 60)
    print("ENHANCED CONVERSION ANALYSIS REPORT")
    print("=" * 60)

    print("\nDocument Overview:")
    print(f"  Total pages: {analysis['total_pages']}")
    print(f"  Total headings found: {len(analysis['headings'])}")
    print(f"  Total captions found: {len(analysis['captions'])}")
    print(f"  Blocks enhanced by LLM: {len(analysis['enhanced_blocks'])}")

    print("\nHeading Level Distribution:")
    for level, count in sorted(analysis["heading_levels"].items()):
        print(f"  Level {level}: {count} headings")

    if analysis["headings"]:
        print("\nSample Headings:")
        for i, heading in enumerate(analysis["headings"][:5]):
            enhanced_marker = " [ENHANCED]" if heading["enhanced"] else ""
            print(
                f"  {i + 1}. Level {heading['level']}: {heading['text'][:60]}...{enhanced_marker}"
            )

    if analysis["captions"]:
        print("\nSample Captions:")
        for i, caption in enumerate(analysis["captions"][:3]):
            enhanced_marker = " [ENHANCED]" if caption["enhanced"] else ""
            print(f"  {i + 1}. {caption['text']}{enhanced_marker}")

    if analysis["enhanced_blocks"]:
        print("\nLLM Enhancement Details:")
        for enhancement in analysis["enhanced_blocks"][:5]:
            print(f"  {enhancement['original_type']} → {enhancement['type']}")
            if enhancement["reasoning"]:
                print(f"    Reasoning: {enhancement['reasoning'][:100]}...")


def main():
    """Main example function."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced PDF conversion example")
    parser.add_argument("pdf_path", help="Path to the PDF file to convert")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM enhancements"
    )
    parser.add_argument(
        "--llm-service",
        choices=["openai", "claude", "gemini", "nvidia"],
        default="gemini",
        help="LLM service to use",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print detailed analysis of the conversion",
    )

    args = parser.parse_args()

    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found")
        return 1

    # Set up output path
    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_path = f"{base_name}_enhanced.md"

    # Configuration overrides for demonstration
    config_overrides = {
        # Heading detection settings
        "heading_min_font_ratio": 1.05,  # Slightly more sensitive
        "heading_max_length": 250,
        # Caption detection settings
        "caption_max_distance": 0.2,  # Allow slightly more distance
        "caption_min_length": 8,
        # LLM settings
        "llm_refinement_confidence": 0.6,  # Lower threshold for more refinements
    }

    try:
        # Convert the document
        document = convert_pdf_with_enhancements(
            pdf_path=args.pdf_path,
            output_path=output_path,
            use_llm=not args.no_llm,
            llm_service=args.llm_service,
            config_overrides=config_overrides,
        )

        # Analyze results if requested
        if args.analyze:
            analysis = analyze_conversion_improvements(document)
            print_analysis_report(analysis)

        print("\nConversion completed successfully!")
        print("Enhanced features used:")
        print("  ✓ Enhanced heading detection")
        print("  ✓ Enhanced caption detection")
        if not args.no_llm:
            print(f"  ✓ LLM-based layout refinement ({args.llm_service})")
            print("  ✓ Layout consistency checking")
        else:
            print("  ✗ LLM enhancements disabled")

        return 0

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


# Additional utility functions for testing and evaluation


def compare_conversions(pdf_path: str, output_dir: str = "comparison_outputs"):
    """
    Compare standard vs enhanced conversion to show improvements.
    """
    import os

    from marker.converters.pdf import PdfConverter

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Standard conversion
    print("Running standard conversion...")
    standard_converter = PdfConverter()
    standard_doc = standard_converter(pdf_path)

    standard_output = os.path.join(output_dir, f"{base_name}_standard.md")
    with open(standard_output, "w", encoding="utf-8") as f:
        f.write(standard_doc.render())

    # Enhanced conversion
    print("Running enhanced conversion...")
    enhanced_doc = convert_pdf_with_enhancements(pdf_path)

    enhanced_output = os.path.join(output_dir, f"{base_name}_enhanced.md")
    with open(enhanced_output, "w", encoding="utf-8") as f:
        f.write(enhanced_doc.render())

    # Analysis
    standard_analysis = analyze_conversion_improvements(standard_doc)
    enhanced_analysis = analyze_conversion_improvements(enhanced_doc)

    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)
    print("Standard conversion:")
    print(f"  Headings: {len(standard_analysis['headings'])}")
    print(f"  Captions: {len(standard_analysis['captions'])}")

    print("\nEnhanced conversion:")
    print(f"  Headings: {len(enhanced_analysis['headings'])}")
    print(f"  Captions: {len(enhanced_analysis['captions'])}")
    print(f"  LLM enhancements: {len(enhanced_analysis['enhanced_blocks'])}")

    improvement_headings = len(enhanced_analysis["headings"]) - len(
        standard_analysis["headings"]
    )
    improvement_captions = len(enhanced_analysis["captions"]) - len(
        standard_analysis["captions"]
    )

    print("\nImprovements:")
    print(f"  Additional headings detected: {improvement_headings}")
    print(f"  Additional captions detected: {improvement_captions}")

    print("\nOutput files:")
    print(f"  Standard: {standard_output}")
    print(f"  Enhanced: {enhanced_output}")


if __name__ == "__main__":
    # If running as script, use the main function
    # If importing, the functions are available for use
    pass
