from typing import Annotated, List, Optional, Tuple

from pydantic import BaseModel

from marker.processors import BaseProcessor, register_processor
from marker.processors.llm import BaseLLMSimpleBlockProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document
from marker.schema.registry import get_block_class


class LayoutRefinementResponse(BaseModel):
    """Response schema for layout refinement."""

    analysis: str
    block_type: str
    confidence: float
    reasoning: str


@register_processor("llm_layout_refinement")
class LLMLayoutRefinementProcessor(BaseLLMSimpleBlockProcessor):
    """
    LLM-based processor for refining layout classification, specifically targeting:
    - Text blocks that should be headings
    - Text blocks that should be captions
    - Improving overall layout accuracy
    """

    block_types: Annotated[
        Tuple[BlockTypes],
        "Block types to analyze for potential reclassification.",
    ] = (BlockTypes.Text, BlockTypes.SectionHeader, BlockTypes.Caption)

    confidence_threshold: Annotated[
        float,
        "Minimum confidence threshold for layout changes.",
    ] = 0.7

    max_text_length: Annotated[
        int,
        "Maximum text length to process with LLM (to avoid token limits).",
    ] = 300

    layout_refinement_prompt: Annotated[
        str,
        "Prompt for layout refinement analysis.",
    ] = """You are an expert in document layout analysis. Your task is to analyze a text block and determine its most appropriate classification.

Look at the provided image of a text block and consider:
1. Font size, weight, and formatting
2. Position and spacing relative to other content
3. Text content and patterns
4. Context within the document layout

Text content: "{text_content}"

Possible classifications:
- **Text**: Regular paragraph or body text
- **SectionHeader**: Section headings, chapter titles, or other hierarchical headers
- **Caption**: Descriptive text for images, tables, or figures

Consider these indicators:
- **SectionHeader**: Larger font, bold formatting, short text, positioned with spacing, hierarchical numbering
- **Caption**: Often smaller font, italic formatting, starts with "Figure/Table/Image X", describes visual content
- **Text**: Regular font size, paragraph-like content, flows naturally with other text

Analyze the image carefully and provide your assessment."""

    def generate_prompt(self, document: Document, page, block: Block) -> str:
        """Generate prompt for layout refinement."""
        text_content = block.raw_text(document).strip()

        # Truncate if too long
        if len(text_content) > self.max_text_length:
            text_content = text_content[: self.max_text_length] + "..."

        return self.layout_refinement_prompt.format(text_content=text_content)

    def get_response_schema(self) -> type[BaseModel]:
        """Return the response schema."""
        return LayoutRefinementResponse

    def should_process_block(self, document: Document, page, block: Block) -> bool:
        """Determine if a block should be processed."""
        # Skip if already processed by this processor
        if hasattr(block, "_llm_layout_processed"):
            return False

        # Skip very long text blocks (likely body text)
        text = block.raw_text(document).strip()
        if len(text) > self.max_text_length:
            return False

        # Skip very short text blocks
        if len(text) < 5:
            return False

        # Process blocks that might be misclassified
        if block.block_type == BlockTypes.Text:
            # Check if it might be a heading or caption
            return self.could_be_heading_or_caption(text, block, document)
        elif block.block_type in (BlockTypes.SectionHeader, BlockTypes.Caption):
            # Check if classification confidence is low
            if hasattr(block, "top_k") and block.top_k:
                confidence = block.top_k.get(block.block_type, 1.0)
                return confidence < 0.8

        return False

    def could_be_heading_or_caption(
        self, text: str, block: Block, document: Document
    ) -> bool:
        """Quick heuristic to check if text could be heading or caption."""
        # Potential heading indicators
        if len(text) < 100 and (
            text.isupper()
            or text.istitle()
            or any(
                pattern in text.lower()
                for pattern in ["chapter", "section", "introduction"]
            )
        ):
            return True

        # Potential caption indicators
        if len(text) < 200 and any(
            pattern in text.lower()
            for pattern in ["figure", "table", "image", "chart", "shows", "depicts"]
        ):
            return True

        return False

    def rewrite_block(self, response: dict, prompt_data: dict, document: Document):
        """Process the LLM response and update block classification if needed."""
        block = prompt_data["block"]
        page = prompt_data["page"]

        if not response:
            return

        new_block_type = response.get("block_type")
        confidence = response.get("confidence", 0.0)
        reasoning = response.get("reasoning", "")

        # Mark as processed to avoid reprocessing
        block._llm_layout_processed = True

        # Only make changes if confidence is high enough
        if confidence < self.confidence_threshold:
            return

        # Validate the new block type
        if new_block_type not in ["Text", "SectionHeader", "Caption"]:
            return

        new_block_type_enum = BlockTypes[new_block_type]

        # Skip if no change needed
        if block.block_type == new_block_type_enum:
            return

        # Convert block to new type
        self.convert_block_type(block, new_block_type_enum, page, document, reasoning)

    def convert_block_type(
        self,
        block: Block,
        new_block_type: BlockTypes,
        page,
        document: Document,
        reasoning: str,
    ):
        """Convert a block to a new type."""
        try:
            new_block_cls = get_block_class(new_block_type)

            # Create new block with same basic properties
            new_block = new_block_cls(
                polygon=block.polygon,
                page_id=block.page_id,
                structure=block.structure,
                source="processor",  # Mark as processor-generated
            )

            # Copy relevant metadata
            if hasattr(block, "metadata") and block.metadata:
                new_block.metadata = block.metadata

            # Add refinement metadata
            new_block.update_metadata(
                llm_layout_refinement=True,
                llm_refinement_reasoning=reasoning,
                original_block_type=str(block.block_type),
            )

            # Special handling for SectionHeader
            if new_block_type == BlockTypes.SectionHeader:
                # Try to determine heading level based on content and formatting
                heading_level = self.estimate_heading_level(block, document)
                new_block.heading_level = heading_level

            # Replace the block
            page.replace_block(block, new_block)

        except Exception as e:
            # Log error but don't fail the entire processing
            print(f"Error converting block type: {e}")

    def estimate_heading_level(self, block: Block, document: Document) -> int:
        """Estimate appropriate heading level for a converted heading."""
        text = block.raw_text(document).strip()

        # Simple heuristics for heading level
        if any(word in text.lower() for word in ["chapter", "part"]):
            return 1
        elif any(
            word in text.lower() for word in ["section", "introduction", "conclusion"]
        ):
            return 2
        elif len(text) < 30 and text.isupper():
            return 1
        elif len(text) < 50:
            return 2
        else:
            return 3

    def extract_image(
        self,
        document: Document,
        image_block: Block,
        remove_blocks: Optional[List[BlockTypes]] = None,
    ) -> "Image.Image":
        """Extract image for the block with some context."""
        # Get a slightly larger area to provide context
        expansion_ratio = 0.05
        return image_block.get_image(
            document,
            highres=True,
            expansion=(expansion_ratio, expansion_ratio),
            remove_blocks=remove_blocks,
        )


# Additional utility processor for post-processing refinements
@register_processor("layout_consistency_checker")
class LayoutConsistencyChecker(BaseProcessor):
    """
    Post-processing checker to ensure layout consistency after refinements.
    """

    block_types = (BlockTypes.SectionHeader, BlockTypes.Caption, BlockTypes.Text)

    def __call__(self, document: Document):
        """Check and fix layout inconsistencies."""
        self.fix_heading_hierarchy(document)
        self.group_related_captions(document)

    def fix_heading_hierarchy(self, document: Document):
        """Ensure heading levels follow a logical hierarchy."""
        headings = []

        # Collect all headings
        for page in document.pages:
            for block in page.contained_blocks(document, (BlockTypes.SectionHeader,)):
                if hasattr(block, "heading_level") and block.heading_level:
                    headings.append(
                        {
                            "block": block,
                            "level": block.heading_level,
                            "page_id": page.page_id,
                            "position": block.polygon.y_start,
                        }
                    )

        # Sort by document order (page, then position)
        headings.sort(key=lambda x: (x["page_id"], x["position"]))

        # Fix hierarchy gaps
        prev_level = 0
        for heading in headings:
            current_level = heading["level"]

            # Don't allow jumps of more than 1 level
            if current_level > prev_level + 1:
                heading["block"].heading_level = prev_level + 1
                current_level = prev_level + 1

            prev_level = current_level

    def group_related_captions(self, document: Document):
        """Group captions with their associated visual elements."""
        # This could be enhanced to create proper figure/table groups
        # For now, we'll just ensure captions are properly positioned

        for page in document.pages:
            captions = page.contained_blocks(document, (BlockTypes.Caption,))
            visuals = page.contained_blocks(
                document, (BlockTypes.Picture, BlockTypes.Figure, BlockTypes.Table)
            )

            # Ensure captions are positioned near their visual elements
            for caption in captions:
                closest_visual = self.find_closest_visual(caption, visuals)
                if closest_visual:
                    # Could implement grouping logic here
                    pass

    def find_closest_visual(
        self, caption: Block, visuals: List[Block]
    ) -> Optional[Block]:
        """Find the closest visual element to a caption."""
        if not visuals:
            return None

        min_distance = float("inf")
        closest_visual = None

        for visual in visuals:
            distance = self.calculate_distance(caption.polygon, visual.polygon)
            if distance < min_distance:
                min_distance = distance
                closest_visual = visual

        return closest_visual

    def calculate_distance(self, poly1, poly2) -> float:
        """Calculate distance between two polygons."""
        # Simple center-to-center distance
        center1 = (poly1.x_start + poly1.width / 2, poly1.y_start + poly1.height / 2)
        center2 = (poly2.x_start + poly2.width / 2, poly2.y_start + poly2.height / 2)

        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
