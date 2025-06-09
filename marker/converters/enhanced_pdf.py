from typing import Annotated, Tuple, Type

from marker.converters import register_converter
from marker.converters.pdf import PdfConverter
from marker.processors import BaseProcessor
from marker.processors.blockquote import BlockquoteProcessor
from marker.processors.code import CodeProcessor
from marker.processors.debug import DebugProcessor
from marker.processors.document_toc import DocumentTOCProcessor
from marker.processors.enhanced_caption_detector import EnhancedCaptionDetectorProcessor

# Import our custom processors
from marker.processors.enhanced_heading_detector import EnhancedHeadingDetectorProcessor
from marker.processors.equation import EquationProcessor
from marker.processors.footnote import FootnoteProcessor
from marker.processors.ignoretext import IgnoreTextProcessor
from marker.processors.line_merge import LineMergeProcessor
from marker.processors.line_numbers import LineNumbersProcessor
from marker.processors.list import ListProcessor
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_equation import LLMEquationProcessor
from marker.processors.llm.llm_form import LLMFormProcessor
from marker.processors.llm.llm_handwriting import LLMHandwritingProcessor
from marker.processors.llm.llm_image_description import LLMImageDescriptionProcessor
from marker.processors.llm.llm_layout_refinement import (
    LayoutConsistencyChecker,
    LLMLayoutRefinementProcessor,
)
from marker.processors.llm.llm_mathblock import LLMMathBlockProcessor
from marker.processors.llm.llm_table import LLMTableProcessor
from marker.processors.llm.llm_table_merge import LLMTableMergeProcessor

# Import existing processors
from marker.processors.order import OrderProcessor
from marker.processors.page_header import PageHeaderProcessor
from marker.processors.reference import ReferenceProcessor
from marker.processors.sectionheader import SectionHeaderProcessor
from marker.processors.table import TableProcessor
from marker.processors.text import TextProcessor


@register_converter("enhanced_pdf")
class EnhancedPdfConverter(PdfConverter):
    """
    Enhanced PDF converter with improved heading detection and caption handling.

    This converter includes:
    - Enhanced heading detection using font analysis and content patterns
    - Improved caption detection with spatial relationship analysis
    - LLM-based layout refinement for better classification
    - Consistency checking for layout hierarchy
    """

    use_enhanced_heading_detection: Annotated[
        bool,
        "Enable enhanced heading detection using font and content analysis.",
    ] = True

    use_enhanced_caption_detection: Annotated[
        bool,
        "Enable enhanced caption detection using spatial and content analysis.",
    ] = True

    use_llm_layout_refinement: Annotated[
        bool,
        "Enable LLM-based layout refinement for better classification.",
    ] = True

    use_layout_consistency_checking: Annotated[
        bool,
        "Enable post-processing consistency checks.",
    ] = True

    # Enhanced processor pipeline
    default_processors: Tuple[Type[BaseProcessor], ...] = (
        # Phase 1: Basic processing and ordering
        OrderProcessor,
        LineMergeProcessor,
        # Phase 2: Content type detection
        BlockquoteProcessor,
        CodeProcessor,
        EquationProcessor,
        FootnoteProcessor,
        IgnoreTextProcessor,
        LineNumbersProcessor,
        ListProcessor,
        PageHeaderProcessor,
        # Phase 3: Enhanced layout detection (our custom processors)
        # Note: These will be conditionally added based on configuration
        # Phase 4: Original section header processing (as fallback)
        SectionHeaderProcessor,
        # Phase 5: Table and form processing
        TableProcessor,
        LLMTableProcessor,
        LLMTableMergeProcessor,
        LLMFormProcessor,
        # Phase 6: Text processing and merging
        TextProcessor,
        # Phase 7: LLM-based enhancements
        LLMComplexRegionProcessor,
        LLMImageDescriptionProcessor,
        LLMEquationProcessor,
        LLMHandwritingProcessor,
        LLMMathBlockProcessor,
        # Phase 8: Final processing
        DocumentTOCProcessor,  # Moved later to capture all headings
        ReferenceProcessor,
        # Phase 9: Debug and cleanup
        DebugProcessor,
    )

    def __init__(self, config=None):
        super().__init__(config)

        # Build the enhanced processor list
        self.processor_list = self.build_enhanced_processor_list()

    def build_enhanced_processor_list(self):
        """Build the processor list with conditional enhancements."""
        processors = []

        # Add processors in phases
        base_processors = list(self.default_processors)

        # Find insertion points
        section_header_idx = None
        text_processor_idx = None
        doc_toc_idx = None

        for i, processor in enumerate(base_processors):
            if processor == SectionHeaderProcessor:
                section_header_idx = i
            elif processor == TextProcessor:
                text_processor_idx = i
            elif processor == DocumentTOCProcessor:
                doc_toc_idx = i

        # Phase 1-2: Basic processing (up to PageHeaderProcessor)
        page_header_idx = None
        for i, processor in enumerate(base_processors):
            if processor == PageHeaderProcessor:
                page_header_idx = i
                break

        if page_header_idx is not None:
            processors.extend(base_processors[: page_header_idx + 1])

        # Phase 3: Enhanced layout detection
        if self.use_enhanced_heading_detection:
            processors.append(EnhancedHeadingDetectorProcessor)

        if self.use_enhanced_caption_detection:
            processors.append(EnhancedCaptionDetectorProcessor)

        # Phase 4: Continue with original processors (SectionHeader as fallback)
        if section_header_idx is not None:
            processors.append(SectionHeaderProcessor)

        # Phase 5-6: Table and text processing
        if text_processor_idx is not None:
            # Add processors from after PageHeader to TextProcessor (inclusive)
            start_idx = page_header_idx + 1 if page_header_idx is not None else 0
            end_idx = text_processor_idx + 1

            # Skip SectionHeaderProcessor if we already added it
            for processor in base_processors[start_idx:end_idx]:
                if processor != SectionHeaderProcessor:
                    processors.append(processor)

        # Phase 7: LLM-based layout refinement
        if self.use_llm_layout_refinement and self.use_llm:
            processors.append(LLMLayoutRefinementProcessor)

        # Phase 8: Continue with LLM enhancements
        if text_processor_idx is not None:
            # Add processors after TextProcessor up to DocumentTOCProcessor
            start_idx = text_processor_idx + 1
            end_idx = doc_toc_idx if doc_toc_idx is not None else len(base_processors)

            for processor in base_processors[start_idx:end_idx]:
                if processor != DocumentTOCProcessor:
                    processors.append(processor)

        # Phase 9: Layout consistency checking
        if self.use_layout_consistency_checking:
            processors.append(LayoutConsistencyChecker)

        # Phase 10: Final processing (DocumentTOC and remaining)
        if doc_toc_idx is not None:
            processors.extend(base_processors[doc_toc_idx:])

        # Instantiate processors with config
        instantiated_processors = []
        for processor_cls in processors:
            try:
                if hasattr(processor_cls, "__init__"):
                    # Check if processor needs special initialization
                    if processor_cls in [
                        EnhancedHeadingDetectorProcessor,
                        EnhancedCaptionDetectorProcessor,
                        LayoutConsistencyChecker,
                    ]:
                        processor = processor_cls(self.config)
                    elif processor_cls == LLMLayoutRefinementProcessor:
                        # LLM processor needs service
                        processor = processor_cls(
                            llm_service=self.llm_service, config=self.config
                        )
                    else:
                        # Use parent class instantiation logic
                        processor = self.resolve_dependencies(processor_cls)
                else:
                    processor = processor_cls

                instantiated_processors.append(processor)
            except Exception as e:
                print(f"Warning: Could not instantiate processor {processor_cls}: {e}")
                # Fall back to parent instantiation
                try:
                    processor = self.resolve_dependencies(processor_cls)
                    instantiated_processors.append(processor)
                except:
                    print(f"Skipping processor {processor_cls}")

        return instantiated_processors

    def get_processor_config(self, processor_name: str) -> dict:
        """Get configuration for a specific processor."""
        if not self.config:
            return {}

        # Look for processor-specific config
        processor_config = {}

        if processor_name == "enhanced_heading_detector":
            processor_config.update(
                {
                    "min_font_size_ratio": getattr(
                        self.config, "heading_min_font_ratio", 1.1
                    ),
                    "max_heading_length": getattr(
                        self.config, "heading_max_length", 200
                    ),
                    "font_weight_threshold": getattr(
                        self.config, "heading_font_weight_threshold", 600.0
                    ),
                }
            )
        elif processor_name == "enhanced_caption_detector":
            processor_config.update(
                {
                    "max_caption_distance": getattr(
                        self.config, "caption_max_distance", 0.15
                    ),
                    "max_caption_length": getattr(
                        self.config, "caption_max_length", 500
                    ),
                    "min_caption_length": getattr(
                        self.config, "caption_min_length", 10
                    ),
                }
            )
        elif processor_name == "llm_layout_refinement":
            processor_config.update(
                {
                    "confidence_threshold": getattr(
                        self.config, "llm_refinement_confidence", 0.7
                    ),
                    "max_text_length": getattr(
                        self.config, "llm_refinement_max_length", 300
                    ),
                }
            )

        return processor_config


# Configuration class for the enhanced converter
class EnhancedPdfConfig:
    """Configuration class for the enhanced PDF converter."""

    def __init__(self):
        # Enhanced heading detection settings
        self.heading_min_font_ratio = 1.1
        self.heading_max_length = 200
        self.heading_font_weight_threshold = 600.0

        # Enhanced caption detection settings
        self.caption_max_distance = 0.15
        self.caption_max_length = 500
        self.caption_min_length = 10

        # LLM refinement settings
        self.llm_refinement_confidence = 0.7
        self.llm_refinement_max_length = 300

        # Feature toggles
        self.use_enhanced_heading_detection = True
        self.use_enhanced_caption_detection = True
        self.use_llm_layout_refinement = True
        self.use_layout_consistency_checking = True

        # Base converter settings
        self.use_llm = True

    def __getitem__(self, key):
        """Allow dictionary-style access for compatibility."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Allow dictionary-style assignment for compatibility."""
        setattr(self, key, value)

    def __contains__(self, key):
        """Allow 'in' operator for compatibility."""
        return hasattr(self, key)

    def get(self, key, default=None):
        """Allow dict.get() style access for compatibility."""
        return getattr(self, key, default)


# Example usage function
def create_enhanced_converter(config_overrides=None):
    """Create an enhanced PDF converter with optional configuration overrides."""
    config = EnhancedPdfConfig()

    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    return EnhancedPdfConverter(config)
