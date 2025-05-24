import html
import re
from typing import Literal, List, cast

import regex

from marker.schema import BlockTypes
from marker.schema.blocks import Block, BlockOutput
from marker.schema.text.span import Span

HYPHENS = r"-—¬"


def remove_tags(text):
    return re.sub(r"<[^>]+>", "", text)


def replace_last(string, old, new):
    matches = list(re.finditer(old, string))
    if not matches:
        return string
    last_match = matches[-1]
    return string[: last_match.start()] + new + string[last_match.end() :]


def strip_trailing_hyphens(line_text, next_line_text, line_html) -> str:
    lowercase_letters = r"\p{Ll}"

    hyphen_regex = regex.compile(rf".*[{HYPHENS}]\s?$", regex.DOTALL)
    next_line_starts_lowercase = regex.match(
        rf"^\s?[{lowercase_letters}]", next_line_text
    )

    if hyphen_regex.match(line_text) and next_line_starts_lowercase:
        line_html = replace_last(line_html, rf"[{HYPHENS}]", "")

    return line_html


class Line(Block):
    block_type: BlockTypes = BlockTypes.Line
    block_description: str = "A line of text."
    formats: List[Literal["math"]] | None = (
        None  # Sometimes we want to set math format at the line level, not span
    )

    def ocr_input_text(self, document):
        text = ""
        for block in self.contained_blocks(document, (BlockTypes.Span,)):
            # Cast to Span since we know these are Span blocks
            span = cast(Span, block)
            # We don't include superscripts/subscripts and math since they can be unreliable at this stage
            block_text = span.text
            if span.italic:
                text += f"<i>{block_text}</i>"
            elif span.bold:
                text += f"<b>{block_text}</b>"
            else:
                text += block_text

        return text.strip()

    def formatted_text(self, document, skip_urls=False):
        text = ""
        for block in self.contained_blocks(document, (BlockTypes.Span,)):
            # Cast to Span since we know these are Span blocks
            span = cast(Span, block)
            block_text = html.escape(span.text)

            if span.has_superscript:
                block_text = re.sub(r"^([0-9\W]+)(.*)", r"<sup>\1</sup>\2", block_text)
                if "<sup>" not in block_text:
                    block_text = f"<sup>{block_text}</sup>"

            if span.url and not skip_urls:
                block_text = f"<a href='{span.url}'>{block_text}</a>"

            if span.italic:
                text += f"<i>{block_text}</i>"
            elif span.bold:
                text += f"<b>{block_text}</b>"
            elif span.math:
                text += f"<math display='inline'>{block_text}</math>"
            else:
                text += block_text

        return text

    def assemble_html(self, document, child_blocks, parent_structure):
        template = ""
        for c in child_blocks:
            template += c.html

        raw_text = remove_tags(template).strip()
        
        # Add None check for parent_structure
        if parent_structure is None:
            return template.strip(" ")
        
        structure_idx = parent_structure.index(str(self.id))
        if structure_idx < len(parent_structure) - 1:
            next_block_id_str = parent_structure[structure_idx + 1]
            # Convert string back to BlockId if needed
            if isinstance(next_block_id_str, str):
                from marker.schema.blocks import BlockId
                import re
                # Parse the string to extract page_id, block_type, block_id
                # Example string: /page/1/Line/2
                m = re.match(r"/page/(\d+)/(\w+)/(\d+)", next_block_id_str)
                if m:
                    page_id, block_type_str, block_id = m.groups()
                    # Convert string to BlockTypes enum
                    try:
                        block_type = BlockTypes[block_type_str]
                        next_block_id = BlockId(page_id=int(page_id), block_type=block_type, block_id=int(block_id))
                    except KeyError:
                        # If the block type is not valid, use fallback
                        next_block_id = next_block_id_str
                else:
                    # fallback: just use page_id
                    m = re.match(r"/page/(\d+)", next_block_id_str)
                    if m:
                        page_id = m.group(1)
                        next_block_id = BlockId(page_id=int(page_id))
                    else:
                        next_block_id = next_block_id_str
            else:
                next_block_id = next_block_id_str
            
            # Only call get_block if we successfully converted to BlockId
            if not isinstance(next_block_id, str):
                next_line = document.get_block(next_block_id)
                if next_line is not None:
                    next_line_raw_text = next_line.raw_text(document)
                    template = strip_trailing_hyphens(raw_text, next_line_raw_text, template)
        else:
            template = template.strip(
                " "
            )  # strip any trailing whitespace from the last line
        return template

    def render(self, document, parent_structure, section_hierarchy=None):
        child_content = []
        if self.structure is not None and len(self.structure) > 0:
            for block_id in self.structure:
                block = document.get_block(block_id)
                if block is not None:
                    child_content.append(
                        block.render(document, parent_structure, section_hierarchy)
                    )

        return BlockOutput(
            html=self.assemble_html(document, child_content, parent_structure),
            polygon=self.polygon,
            id=self.id,
            children=[],
            section_hierarchy=section_hierarchy,
        )

    def merge(self, other: "Line"):
        self.polygon = self.polygon.merge([other.polygon])

        # Handle merging structure with Nones
        if self.structure is None:
            self.structure = other.structure
        elif other.structure is not None:
            self.structure = self.structure + other.structure

        # Merge formats with Nones
        if self.formats is None:
            self.formats = other.formats
        elif other.formats is not None:
            self.formats = list(set(self.formats + other.formats))
