from typing import Annotated, List, Tuple

from bs4 import BeautifulSoup
from PIL import Image
from marker.logger import get_logger
from pydantic import BaseModel

from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block, TableCell, Table
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox

logger = get_logger()


class OptimizedLLMTableProcessor(BaseLLMComplexBlockProcessor):
    """
    Optimized table processor with better chunking and error handling
    """
    
    block_types: Annotated[
        Tuple[BlockTypes],
        "The block types to process.",
    ] = (BlockTypes.Table, BlockTypes.TableOfContents)
    
    # Reduced batch sizes for better LLM handling
    max_rows_per_batch: Annotated[
        int,
        "Smaller batches for better LLM processing.",
    ] = 15
    max_table_rows: Annotated[
        int,
        "Reduced maximum rows to prevent large payloads.",
    ] = 40
    max_cells_per_batch: Annotated[
        int,
        "Maximum number of cells per batch to control payload size.",
    ] = 50
    
    table_image_expansion_ratio: Annotated[
        float,
        "The ratio to expand the image by when cropping.",
    ] = 0
    rotation_max_wh_ratio: Annotated[
        float,
        "The maximum width/height ratio for table cells for a table to be considered rotated.",
    ] = 0.6
    
    # Improved prompt with clearer instructions
    table_rewriting_prompt: Annotated[
        str,
        "Optimized prompt for better text correction.",
    ] = """You are a text correction expert. You will receive a table image and its HTML representation.

Your task: Correct any errors in the HTML to match the image exactly.

Guidelines:
- Fix OCR errors and misread characters
- Ensure column headers align with correct data
- Clean up any \\n, \\t, or escaped characters
- Use <math> tags for inline math, <math display="block"> for block math
- Replace images with "Image: [description]"
- Use only: th, td, tr, br, span, sup, sub, i, b, math, table tags
- Use only: display, style, colspan, rowspan attributes when necessary
- Use <br> for line breaks within cells

Instructions:
1. Compare the image with the HTML representation
2. Identify any discrepancies or errors
3. If no corrections needed, output: "No corrections needed."
4. If corrections needed, output the corrected HTML

Input HTML:
```html
{block_html}
```

Output the corrected HTML or "No corrections needed.":"""

    def should_process_table(self, block: Table, children: List[TableCell]) -> bool:
        """Determine if table should be processed based on size and complexity"""
        if not children:
            return False
            
        row_count = len(set(cell.row_id for cell in children))
        col_count = len(set(cell.col_id for cell in children))
        total_cells = len(children)
        
        # Skip very large tables that would cause payload issues
        if total_cells > 100 or row_count > self.max_table_rows:
            logger.info(f"Skipping large table: {row_count} rows, {total_cells} cells")
            return False
            
        # Skip tables with very complex structure
        if any(cell.rowspan > 5 or cell.colspan > 5 for cell in children):
            logger.info("Skipping table with complex spans")
            return False
            
        return True

    def chunk_table_cells(self, children: List[TableCell], row_idxs: List[int]) -> List[List[int]]:
        """Create optimized chunks based on both row count and cell count"""
        chunks = []
        current_chunk = []
        current_cell_count = 0
        
        for i in range(0, len(row_idxs), self.max_rows_per_batch):
            batch_row_idxs = row_idxs[i : i + self.max_rows_per_batch]
            batch_cells = [cell for cell in children if cell.row_id in batch_row_idxs]
            
            # If this batch would exceed cell limit, process what we have and start fresh
            if current_cell_count + len(batch_cells) > self.max_cells_per_batch and current_chunk:
                chunks.append(current_chunk)
                current_chunk = batch_row_idxs
                current_cell_count = len(batch_cells)
            else:
                current_chunk.extend(batch_row_idxs)
                current_cell_count += len(batch_cells)
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def clean_cell_text(self, text: str) -> str:
        """Clean up common text issues including escaped characters"""
        if not text:
            return text
            
        # Remove common escape sequences
        text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        
        # Clean up multiple spaces
        while '  ' in text:
            text = text.replace('  ', ' ')
            
        return text.strip()

    def process_rewriting(self, document: Document, page: PageGroup, block: Table):
        children: List[TableCell] = block.contained_blocks(
            document, (BlockTypes.TableCell,)
        )
        
        if not self.should_process_table(block, children):
            return

        # Clean existing cell text first
        for cell in children:
            if cell.text_lines:
                cell.text_lines = [self.clean_cell_text(line) for line in cell.text_lines]

        unique_rows = set([cell.row_id for cell in children])
        row_count = len(unique_rows)
        row_idxs = sorted(list(unique_rows))

        # Create optimized chunks
        row_chunks = self.chunk_table_cells(children, row_idxs)
        
        if len(row_chunks) > 5:  # Too many chunks, skip processing
            logger.info(f"Skipping table with {len(row_chunks)} chunks (too many)")
            return

        parsed_cells = []
        row_shift = 0
        block_image = self.extract_image(document, block)
        
        if not block_image:
            logger.warning("Could not extract block image")
            return
            
        block_rescaled_bbox = block.polygon.rescale(
            page.polygon.size, page.get_image(highres=True).size
        ).bbox

        for chunk_idx, chunk_row_idxs in enumerate(row_chunks):
            try:
                batch_cells = [cell for cell in children if cell.row_id in chunk_row_idxs]
                
                if not batch_cells:
                    continue
                    
                batch_cell_bboxes = [
                    cell.polygon.rescale(
                        page.polygon.size, page.get_image(highres=True).size
                    ).bbox
                    for cell in batch_cells
                ]
                
                # Calculate batch bbox relative to the block
                batch_bbox = [
                    min([bbox[0] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[0],
                    min([bbox[1] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[1],
                    max([bbox[2] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[0],
                    max([bbox[3] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[1],
                ]
                
                # Ensure reasonable crop dimensions
                batch_bbox[0] = max(0, batch_bbox[0])
                batch_bbox[1] = max(0, batch_bbox[1])
                batch_bbox[2] = min(block_image.size[0], batch_bbox[2])
                batch_bbox[3] = min(block_image.size[1], batch_bbox[3])
                
                if batch_bbox[2] <= batch_bbox[0] or batch_bbox[3] <= batch_bbox[1]:
                    logger.warning(f"Invalid batch bbox: {batch_bbox}")
                    continue

                batch_image = block_image.crop(batch_bbox)
                block_html = block.format_cells(document, [], batch_cells)
                
                # Limit HTML length to prevent payload issues
                if len(block_html) > 10000:
                    logger.warning(f"HTML too long ({len(block_html)} chars), skipping batch")
                    continue
                
                batch_image = self.handle_image_rotation(batch_cells, batch_image)
                batch_parsed_cells = self.rewrite_single_chunk(
                    page, block, block_html, batch_cells, batch_image
                )
                
                if batch_parsed_cells is None:
                    logger.warning(f"Failed to process batch {chunk_idx}")
                    # Use original cells for this batch
                    for cell in batch_cells:
                        cell.row_id += row_shift
                        parsed_cells.append(cell)
                else:
                    for cell in batch_parsed_cells:
                        cell.row_id += row_shift
                        parsed_cells.append(cell)
                        
                if batch_parsed_cells:
                    row_shift += max([cell.row_id for cell in batch_parsed_cells])
                else:
                    row_shift += len(set(cell.row_id for cell in batch_cells))
                    
            except Exception as e:
                logger.error(f"Error processing batch {chunk_idx}: {e}")
                # Use original cells for this batch
                for cell in batch_cells:
                    cell.row_id += row_shift
                    parsed_cells.append(cell)
                row_shift += len(set(cell.row_id for cell in batch_cells))

        if parsed_cells:
            block.structure = []
            for cell in parsed_cells:
                page.add_full_block(cell)
                block.add_structure(cell)

    def rewrite_single_chunk(
        self,
        page: PageGroup,
        block: Block,
        block_html: str,
        children: List[TableCell],
        image: Image.Image,
    ):
        prompt = self.table_rewriting_prompt.replace("{block_html}", block_html)

        try:
            response = self.llm_service(prompt, image, block, TableSchema)
        except Exception as e:
            logger.error(f"LLM service error: {e}")
            block.update_metadata(llm_error_count=1)
            return None

        if not response or "corrected_html" not in response:
            block.update_metadata(llm_error_count=1)
            return None

        corrected_html = response["corrected_html"]

        # The original table is okay
        if "no corrections" in corrected_html.lower():
            return children  # Return original cells

        corrected_html = corrected_html.strip().lstrip("```html").rstrip("```").strip()
        
        try:
            parsed_cells = self.parse_html_table(corrected_html, block, page)
        except Exception as e:
            logger.error(f"Error parsing HTML table: {e}")
            block.update_metadata(llm_error_count=1)
            return None
            
        if len(parsed_cells) <= 1:
            block.update_metadata(llm_error_count=1)
            return None

        if not corrected_html.endswith("</table>"):
            block.update_metadata(llm_error_count=1)
            return None

        # Validate the response
        parsed_cell_text = "".join([cell.text for cell in parsed_cells])
        orig_cell_text = "".join([cell.text for cell in children])
        
        # More lenient validation
        if len(parsed_cell_text) < len(orig_cell_text) * 0.3:
            logger.warning("Parsed text too short, using original")
            block.update_metadata(llm_error_count=1)
            return children

        return parsed_cells

    @staticmethod
    def get_cell_text(element, keep_tags=("br", "i", "b", "span", "math")) -> str:
        for tag in element.find_all(True):
            if tag.name not in keep_tags:
                tag.unwrap()
        return element.decode_contents()

    def parse_html_table(
        self, html_text: str, block: Block, page: PageGroup
    ) -> List[TableCell]:
        soup = BeautifulSoup(html_text, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        rows = table.find_all("tr")
        cells = []

        # Find maximum number of columns
        max_cols = 0
        for row in rows:
            row_tds = row.find_all(["td", "th"])
            curr_cols = sum(int(cell.get("colspan", 1)) for cell in row_tds)
            max_cols = max(max_cols, curr_cols)

        grid = [[True] * max_cols for _ in range(len(rows))]

        for i, row in enumerate(rows):
            cur_col = 0
            row_cells = row.find_all(["td", "th"])
            for j, cell in enumerate(row_cells):
                while cur_col < max_cols and not grid[i][cur_col]:
                    cur_col += 1

                if cur_col >= max_cols:
                    break

                cell_text = self.get_cell_text(cell).strip()
                # Clean the cell text
                cell_text = self.clean_cell_text(cell_text)
                
                rowspan = min(int(cell.get("rowspan", 1)), len(rows) - i)
                colspan = min(int(cell.get("colspan", 1)), max_cols - cur_col)
                
                if colspan <= 0 or rowspan <= 0:
                    continue

                # Mark grid cells as occupied
                for r in range(i, i + rowspan):
                    for c in range(cur_col, cur_col + colspan):
                        if r < len(grid) and c < len(grid[r]):
                            grid[r][c] = False

                cell_bbox = [
                    block.polygon.bbox[0] + cur_col,
                    block.polygon.bbox[1] + i,
                    block.polygon.bbox[0] + cur_col + colspan,
                    block.polygon.bbox[1] + i + rowspan,
                ]
                cell_polygon = PolygonBox.from_bbox(cell_bbox)

                cell_obj = TableCell(
                    text_lines=[cell_text] if cell_text else [],
                    row_id=i,
                    col_id=cur_col,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=cell.name == "th",
                    polygon=cell_polygon,
                    page_id=page.page_id,
                )
                cells.append(cell_obj)
                cur_col += colspan

        return cells

    def handle_image_rotation(self, children: List[TableCell], image: Image.Image):
        """Same as original but with error handling"""
        try:
            ratios = [c.polygon.width / c.polygon.height for c in children if c.polygon.height > 0]
            if len(ratios) < 2:
                return image

            is_rotated = all([r < self.rotation_max_wh_ratio for r in ratios])
            if not is_rotated:
                return image

            first_col_id = min([c.col_id for c in children])
            last_col_id = max([c.col_id for c in children])
            
            if last_col_id == first_col_id:
                return image

            first_col_cells = [c for c in children if c.col_id == first_col_id]
            last_col_cells = [c for c in children if c.col_id == last_col_id]
            
            if not first_col_cells or not last_col_cells:
                return image
                
            cell_diff = first_col_cells[0].polygon.y_start - last_col_cells[0].polygon.y_start
            
            if cell_diff == 0:
                return image
            elif cell_diff > 0:
                return image.rotate(270, expand=True)
            else:
                return image.rotate(90, expand=True)
        except Exception as e:
            logger.warning(f"Error in image rotation: {e}")
            return image


class TableSchema(BaseModel):
    comparison: str
    corrected_html: str