from typing import Annotated, Tuple, Dict, List
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import re

from marker.processors import BaseProcessor, register_processor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.blocks import Block
from marker.schema.text.line import Line
from marker.schema.text.span import Span
from marker.schema.registry import get_block_class


@register_processor('enhanced_heading_detector')
class EnhancedHeadingDetectorProcessor(BaseProcessor):
    """
    Enhanced processor for detecting and classifying section headers using multiple signals:
    - Font size and weight
    - Text formatting (bold, italic)
    - Position and spacing
    - Content patterns
    - Context analysis
    """
    
    block_types: Annotated[
        Tuple[BlockTypes],
        "Block types to analyze for potential headings.",
    ] = (BlockTypes.Text, BlockTypes.SectionHeader)
    
    min_font_size_ratio: Annotated[
        float,
        "Minimum font size ratio compared to body text to consider as heading.",
    ] = 1.1
    
    max_heading_length: Annotated[
        int,
        "Maximum character length for text to be considered a heading.",
    ] = 200
    
    heading_patterns: Annotated[
        List[str],
        "Regex patterns that indicate heading content.",
    ] = [
        r'^\d+\.?\s+[A-Z]',  # "1. Introduction", "1 Introduction"
        r'^[A-Z][A-Z\s]{2,}$',  # "INTRODUCTION", "CHAPTER ONE"
        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # "Introduction", "Chapter One"
        r'^\d+\.\d+\.?\s+[A-Z]',  # "1.1 Overview", "1.1. Overview"
        r'^[IVX]+\.?\s+[A-Z]',  # "I. Introduction", "IV Overview"
    ]
    
    position_weight: Annotated[
        float,
        "Weight for position-based heading detection (left alignment, spacing).",
    ] = 0.3
    
    font_weight_threshold: Annotated[
        float,
        "Minimum font weight to consider as potential heading.",
    ] = 600.0

    def __call__(self, document: Document):
        """Main processing method."""
        # Collect font and formatting statistics
        font_stats = self.analyze_document_fonts(document)
        
        # Analyze potential headings
        heading_candidates = self.find_heading_candidates(document, font_stats)
        
        # Classify and assign heading levels
        self.classify_and_assign_headings(document, heading_candidates, font_stats)

    def analyze_document_fonts(self, document: Document) -> Dict:
        """Analyze font usage across the document to establish baselines."""
        font_sizes = []
        font_weights = []
        font_families = []
        
        for page in document.pages:
            for block in page.contained_blocks(document, (BlockTypes.Text, BlockTypes.SectionHeader)):
                for line in block.contained_blocks(document, (BlockTypes.Line,)):
                    for span in line.contained_blocks(document, (BlockTypes.Span,)):
                        font_sizes.append(span.font_size)
                        font_weights.append(span.font_weight)
                        font_families.append(span.font)
        
        if not font_sizes:
            return {
                'body_font_size': 12.0,
                'body_font_weight': 400.0,
                'common_font_family': 'default'
            }
        
        return {
            'body_font_size': np.median(font_sizes),
            'body_font_weight': np.median(font_weights),
            'common_font_family': Counter(font_families).most_common(1)[0][0],
            'font_size_std': np.std(font_sizes),
            'all_font_sizes': font_sizes
        }

    def find_heading_candidates(self, document: Document, font_stats: Dict) -> List[Dict]:
        """Find blocks that could potentially be headings."""
        candidates = []
        
        for page in document.pages:
            for block in page.contained_blocks(document, self.block_types):
                if not block.structure:
                    continue
                
                # Get text content and formatting
                text = block.raw_text(document).strip()
                if not text or len(text) > self.max_heading_length:
                    continue
                
                # Analyze formatting
                formatting_score = self.analyze_block_formatting(block, document, font_stats)
                
                # Analyze content patterns
                content_score = self.analyze_content_patterns(text)
                
                # Analyze position and spacing
                position_score = self.analyze_position_context(block, page, document)
                
                # Calculate overall heading probability
                total_score = (
                    formatting_score * 0.5 +
                    content_score * 0.3 +
                    position_score * self.position_weight
                )
                
                if total_score > 0.4:  # Threshold for considering as heading candidate
                    candidates.append({
                        'block': block,
                        'text': text,
                        'score': total_score,
                        'formatting_score': formatting_score,
                        'content_score': content_score,
                        'position_score': position_score,
                        'font_size': self.get_dominant_font_size(block, document),
                        'font_weight': self.get_dominant_font_weight(block, document),
                        'is_bold': self.is_predominantly_bold(block, document),
                        'page_id': page.page_id
                    })
        
        return candidates

    def analyze_block_formatting(self, block: Block, document: Document, font_stats: Dict) -> float:
        """Analyze font size, weight, and formatting to determine heading likelihood."""
        score = 0.0
        
        # Get dominant font characteristics
        font_size = self.get_dominant_font_size(block, document)
        font_weight = self.get_dominant_font_weight(block, document)
        
        # Font size score
        size_ratio = font_size / font_stats['body_font_size']
        if size_ratio >= self.min_font_size_ratio:
            score += min(0.4, (size_ratio - 1.0) * 0.4)
        
        # Font weight score
        if font_weight >= self.font_weight_threshold:
            score += 0.3
        
        # Bold formatting score
        if self.is_predominantly_bold(block, document):
            score += 0.2
        
        # All caps bonus
        text = block.raw_text(document).strip()
        if text.isupper() and len(text) > 3:
            score += 0.1
        
        return min(1.0, score)

    def analyze_content_patterns(self, text: str) -> float:
        """Analyze text content for heading-like patterns."""
        score = 0.0
        
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                score += 0.3
                break
        
        # Short, title-case text
        if len(text) < 50 and text.istitle():
            score += 0.2
        
        # Ends without punctuation (except colon)
        if not text.endswith(('.', '!', '?', ';')) or text.endswith(':'):
            score += 0.1
        
        # Contains common heading words
        heading_words = ['chapter', 'section', 'introduction', 'conclusion', 
                        'overview', 'summary', 'abstract', 'background', 'methodology']
        if any(word in text.lower() for word in heading_words):
            score += 0.2
        
        return min(1.0, score)

    def analyze_position_context(self, block: Block, page, document: Document) -> float:
        """Analyze block position and spacing context."""
        score = 0.0
        
        # Left alignment (assuming left-aligned headings)
        page_left_margin = page.polygon.width * 0.1
        if block.polygon.x_start <= page_left_margin:
            score += 0.3
        
        # Spacing above and below
        prev_block = page.get_prev_block(block)
        next_block = page.get_next_block(block)
        
        if prev_block:
            gap_above = block.polygon.y_start - prev_block.polygon.y_end
            if gap_above > block.polygon.height * 0.5:  # Significant gap above
                score += 0.2
        
        if next_block:
            gap_below = next_block.polygon.y_start - block.polygon.y_end
            if gap_below > block.polygon.height * 0.3:  # Some gap below
                score += 0.1
        
        # Single line blocks are more likely to be headings
        lines = block.contained_blocks(document, (BlockTypes.Line,))
        if len(lines) == 1:
            score += 0.2
        
        return min(1.0, score)

    def get_dominant_font_size(self, block: Block, document: Document) -> float:
        """Get the most common font size in the block."""
        font_sizes = []
        for line in block.contained_blocks(document, (BlockTypes.Line,)):
            for span in line.contained_blocks(document, (BlockTypes.Span,)):
                font_sizes.append(span.font_size)
        
        return np.median(font_sizes) if font_sizes else 12.0

    def get_dominant_font_weight(self, block: Block, document: Document) -> float:
        """Get the most common font weight in the block."""
        font_weights = []
        for line in block.contained_blocks(document, (BlockTypes.Line,)):
            for span in line.contained_blocks(document, (BlockTypes.Span,)):
                font_weights.append(span.font_weight)
        
        return np.median(font_weights) if font_weights else 400.0

    def is_predominantly_bold(self, block: Block, document: Document) -> bool:
        """Check if the majority of text in the block is bold."""
        bold_chars = 0
        total_chars = 0
        
        for line in block.contained_blocks(document, (BlockTypes.Line,)):
            for span in line.contained_blocks(document, (BlockTypes.Span,)):
                char_count = len(span.text)
                total_chars += char_count
                if span.bold:
                    bold_chars += char_count
        
        return total_chars > 0 and (bold_chars / total_chars) > 0.5

    def classify_and_assign_headings(self, document: Document, candidates: List[Dict], font_stats: Dict):
        """Classify heading candidates and assign appropriate heading levels."""
        if not candidates:
            return
        
        # Sort candidates by font size (descending) and score
        candidates.sort(key=lambda x: (x['font_size'], x['score']), reverse=True)
        
        # Group by font size to determine heading levels
        font_size_groups = defaultdict(list)
        for candidate in candidates:
            font_size_groups[candidate['font_size']].append(candidate)
        
        # Assign heading levels based on font size hierarchy
        sorted_font_sizes = sorted(font_size_groups.keys(), reverse=True)
        
        for level, font_size in enumerate(sorted_font_sizes[:6], 1):  # Max 6 heading levels
            for candidate in font_size_groups[font_size]:
                self.convert_to_heading(candidate['block'], level, document)

    def convert_to_heading(self, block: Block, heading_level: int, document: Document):
        """Convert a block to a SectionHeader with the specified level."""
        if block.block_type != BlockTypes.SectionHeader:
            # Convert Text block to SectionHeader
            page = document.get_page(block.page_id)
            section_header_cls = get_block_class(BlockTypes.SectionHeader)
            
            new_header = section_header_cls(
                polygon=block.polygon,
                page_id=block.page_id,
                structure=block.structure,
                heading_level=heading_level
            )
            
            page.replace_block(block, new_header)
        else:
            # Update existing SectionHeader
            block.heading_level = heading_level
