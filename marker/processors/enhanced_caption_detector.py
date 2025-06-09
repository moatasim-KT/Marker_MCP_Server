from typing import Annotated, Tuple, List, Dict, Optional
import re
import numpy as np
from collections import defaultdict

from marker.processors import BaseProcessor, register_processor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.blocks import Block
from marker.schema.registry import get_block_class


@register_processor('enhanced_caption_detector')
class EnhancedCaptionDetectorProcessor(BaseProcessor):
    """
    Enhanced processor for detecting and classifying captions using multiple signals:
    - Spatial relationship to visual elements (images, tables, figures)
    - Text content patterns (Figure X, Table Y, etc.)
    - Font size and formatting differences
    - Position relative to visual content
    """
    
    block_types: Annotated[
        Tuple[BlockTypes],
        "Block types to analyze for potential captions.",
    ] = (BlockTypes.Text, BlockTypes.Caption)
    
    max_caption_distance: Annotated[
        float,
        "Maximum relative distance from visual element to consider as caption (as fraction of page height).",
    ] = 0.15
    
    max_caption_length: Annotated[
        int,
        "Maximum character length for text to be considered a caption.",
    ] = 500
    
    min_caption_length: Annotated[
        int,
        "Minimum character length for text to be considered a caption.",
    ] = 10
    
    caption_patterns: Annotated[
        List[str],
        "Regex patterns that indicate caption content.",
    ] = [
        r'^(Figure|Fig\.?)\s*\d+',  # "Figure 1", "Fig. 2"
        r'^(Table|Tbl\.?)\s*\d+',   # "Table 1", "Tbl. 2"
        r'^(Image|Img\.?)\s*\d+',   # "Image 1", "Img. 2"
        r'^(Chart|Graph)\s*\d+',    # "Chart 1", "Graph 2"
        r'^(Diagram|Diag\.?)\s*\d+', # "Diagram 1", "Diag. 2"
        r'^(Photo|Photograph)\s*\d+', # "Photo 1", "Photograph 2"
        r'^\d+\.\s*(Figure|Table|Image)', # "1. Figure", "2. Table"
    ]
    
    caption_keywords: Annotated[
        List[str],
        "Keywords commonly found in captions.",
    ] = [
        'shows', 'depicts', 'illustrates', 'displays', 'presents',
        'above', 'below', 'left', 'right', 'source', 'adapted',
        'comparison', 'overview', 'summary', 'example', 'sample'
    ]

    def __call__(self, document: Document):
        """Main processing method."""
        # Find all visual elements (images, tables, figures)
        visual_elements = self.find_visual_elements(document)
        
        # Find potential caption candidates
        caption_candidates = self.find_caption_candidates(document)
        
        # Match captions to visual elements
        caption_matches = self.match_captions_to_visuals(caption_candidates, visual_elements, document)
        
        # Convert matched text blocks to captions
        self.convert_to_captions(caption_matches, document)

    def find_visual_elements(self, document: Document) -> List[Dict]:
        """Find all visual elements that might have captions."""
        visual_elements = []
        
        visual_block_types = (
            BlockTypes.Picture, BlockTypes.Figure, BlockTypes.Table,
            BlockTypes.Image, BlockTypes.Chart
        )
        
        for page in document.pages:
            for block in page.contained_blocks(document, visual_block_types):
                visual_elements.append({
                    'block': block,
                    'type': block.block_type,
                    'page_id': page.page_id,
                    'bbox': block.polygon.bbox,
                    'center_x': block.polygon.x_start + block.polygon.width / 2,
                    'center_y': block.polygon.y_start + block.polygon.height / 2,
                    'area': block.polygon.width * block.polygon.height
                })
        
        return visual_elements

    def find_caption_candidates(self, document: Document) -> List[Dict]:
        """Find text blocks that could potentially be captions."""
        candidates = []
        
        for page in document.pages:
            for block in page.contained_blocks(document, self.block_types):
                if not block.structure:
                    continue
                
                text = block.raw_text(document).strip()
                if (not text or 
                    len(text) < self.min_caption_length or 
                    len(text) > self.max_caption_length):
                    continue
                
                # Analyze content for caption patterns
                content_score = self.analyze_caption_content(text)
                
                # Analyze formatting (captions often have different formatting)
                formatting_score = self.analyze_caption_formatting(block, document)
                
                # Calculate overall caption probability
                total_score = content_score * 0.7 + formatting_score * 0.3
                
                if total_score > 0.3:  # Threshold for considering as caption candidate
                    candidates.append({
                        'block': block,
                        'text': text,
                        'score': total_score,
                        'content_score': content_score,
                        'formatting_score': formatting_score,
                        'page_id': page.page_id,
                        'bbox': block.polygon.bbox,
                        'center_y': block.polygon.y_start + block.polygon.height / 2
                    })
        
        return candidates

    def analyze_caption_content(self, text: str) -> float:
        """Analyze text content for caption-like patterns."""
        score = 0.0
        
        # Check for explicit caption patterns
        for pattern in self.caption_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                score += 0.6
                break
        
        # Check for caption keywords
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in self.caption_keywords if keyword in text_lower)
        if keyword_count > 0:
            score += min(0.3, keyword_count * 0.1)
        
        # Short, descriptive text
        if 10 <= len(text) <= 100:
            score += 0.2
        elif 100 < len(text) <= 200:
            score += 0.1
        
        # Starts with capital letter and ends with period
        if text[0].isupper() and text.endswith('.'):
            score += 0.1
        
        return min(1.0, score)

    def analyze_caption_formatting(self, block: Block, document: Document) -> float:
        """Analyze formatting characteristics typical of captions."""
        score = 0.0
        
        # Get font characteristics
        font_sizes = []
        is_italic_count = 0
        total_spans = 0
        
        for line in block.contained_blocks(document, (BlockTypes.Line,)):
            for span in line.contained_blocks(document, (BlockTypes.Span,)):
                font_sizes.append(span.font_size)
                if span.italic:
                    is_italic_count += 1
                total_spans += 1
        
        if font_sizes:
            avg_font_size = np.mean(font_sizes)
            
            # Captions often have smaller font size
            if avg_font_size < 11:
                score += 0.3
            elif avg_font_size < 12:
                score += 0.1
        
        # Captions often use italic formatting
        if total_spans > 0 and (is_italic_count / total_spans) > 0.5:
            score += 0.2
        
        # Single line blocks are more likely to be captions
        lines = block.contained_blocks(document, (BlockTypes.Line,))
        if len(lines) == 1:
            score += 0.1
        elif len(lines) <= 3:
            score += 0.05
        
        return min(1.0, score)

    def match_captions_to_visuals(self, caption_candidates: List[Dict], 
                                 visual_elements: List[Dict], document: Document) -> List[Dict]:
        """Match caption candidates to their corresponding visual elements."""
        matches = []
        
        for candidate in caption_candidates:
            best_match = None
            min_distance = float('inf')
            
            # Find the closest visual element on the same page
            page_visuals = [v for v in visual_elements if v['page_id'] == candidate['page_id']]
            
            for visual in page_visuals:
                distance = self.calculate_caption_visual_distance(candidate, visual, document)
                
                # Check if within reasonable distance
                page = document.get_page(candidate['page_id'])
                max_distance = page.polygon.height * self.max_caption_distance
                
                if distance < max_distance and distance < min_distance:
                    min_distance = distance
                    best_match = visual
            
            if best_match:
                # Determine caption position relative to visual
                position = self.determine_caption_position(candidate, best_match)
                
                matches.append({
                    'caption_candidate': candidate,
                    'visual_element': best_match,
                    'distance': min_distance,
                    'position': position,
                    'confidence': self.calculate_match_confidence(candidate, best_match, min_distance)
                })
        
        # Filter matches by confidence and resolve conflicts
        return self.resolve_caption_conflicts(matches)

    def calculate_caption_visual_distance(self, caption: Dict, visual: Dict, document: Document) -> float:
        """Calculate distance between caption and visual element."""
        # Use center-to-center distance with bias for vertical proximity
        caption_center_x = caption['bbox'][0] + (caption['bbox'][2] - caption['bbox'][0]) / 2
        caption_center_y = caption['bbox'][1] + (caption['bbox'][3] - caption['bbox'][1]) / 2
        
        visual_center_x = visual['center_x']
        visual_center_y = visual['center_y']
        
        # Calculate distance with higher weight for vertical separation
        dx = abs(caption_center_x - visual_center_x)
        dy = abs(caption_center_y - visual_center_y)
        
        # Weight vertical distance more heavily
        return (dx**2 + (dy * 2)**2)**0.5

    def determine_caption_position(self, caption: Dict, visual: Dict) -> str:
        """Determine if caption is above, below, left, or right of visual."""
        caption_center_y = caption['bbox'][1] + (caption['bbox'][3] - caption['bbox'][1]) / 2
        visual_center_y = visual['center_y']
        
        if caption_center_y < visual_center_y - visual['block'].polygon.height * 0.1:
            return 'above'
        elif caption_center_y > visual_center_y + visual['block'].polygon.height * 0.1:
            return 'below'
        else:
            caption_center_x = caption['bbox'][0] + (caption['bbox'][2] - caption['bbox'][0]) / 2
            if caption_center_x < visual['center_x']:
                return 'left'
            else:
                return 'right'

    def calculate_match_confidence(self, caption: Dict, visual: Dict, distance: float) -> float:
        """Calculate confidence score for caption-visual match."""
        # Base confidence from caption content score
        confidence = caption['score']
        
        # Boost confidence for explicit caption patterns
        if any(re.match(pattern, caption['text'], re.IGNORECASE) for pattern in self.caption_patterns):
            confidence += 0.2
        
        # Reduce confidence based on distance
        # Closer captions get higher confidence
        distance_penalty = min(0.3, distance / 100)  # Normalize distance
        confidence -= distance_penalty
        
        return max(0.0, min(1.0, confidence))

    def resolve_caption_conflicts(self, matches: List[Dict]) -> List[Dict]:
        """Resolve conflicts where multiple captions match the same visual or vice versa."""
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        used_captions = set()
        used_visuals = set()
        resolved_matches = []
        
        for match in matches:
            caption_id = id(match['caption_candidate']['block'])
            visual_id = id(match['visual_element']['block'])
            
            # Only accept if both caption and visual are unused and confidence is high enough
            if (caption_id not in used_captions and 
                visual_id not in used_visuals and 
                match['confidence'] > 0.5):
                
                resolved_matches.append(match)
                used_captions.add(caption_id)
                used_visuals.add(visual_id)
        
        return resolved_matches

    def convert_to_captions(self, matches: List[Dict], document: Document):
        """Convert matched text blocks to caption blocks."""
        for match in matches:
            caption_block = match['caption_candidate']['block']
            
            if caption_block.block_type != BlockTypes.Caption:
                # Convert Text block to Caption
                page = document.get_page(caption_block.page_id)
                caption_cls = get_block_class(BlockTypes.Caption)
                
                new_caption = caption_cls(
                    polygon=caption_block.polygon,
                    page_id=caption_block.page_id,
                    structure=caption_block.structure
                )
                
                # Add metadata about the visual element it describes
                new_caption.update_metadata(
                    visual_element_type=str(match['visual_element']['type']),
                    caption_position=match['position'],
                    match_confidence=match['confidence'],
                    enhanced_caption_detection=True
                )
                
                page.replace_block(caption_block, new_caption)
