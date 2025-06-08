import contextlib
import ctypes
import logging
import re
from typing import Annotated, Dict, List, Optional, Set

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from ftfy import fix_text
from pdftext.extraction import dictionary_output
# Use the marker Reference class for type hints
from marker.schema.blocks.reference import Reference
# Temporary workaround for missing flatten function
try:
    from pdftext.pdf.utils import flatten as flatten_pdf_page
except ImportError:
    # Define a minimal flatten function as fallback
    def flatten_pdf_page(page):  # type: ignore
        """Fallback flatten function that does nothing."""
        _ = page  # Suppress unused parameter warning
        pass
from pydantic import BaseModel

from PIL import Image
from pypdfium2 import PdfiumError, PdfDocument

from marker.providers import BaseProvider, ProviderOutput, Char, ProviderPageLines
from marker.providers.utils import alphanum_ratio
from marker.schema import BlockTypes
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class
from marker.schema.text.span import Span

# Ignore pypdfium2 warning about form flattening
logging.getLogger("pypdfium2").setLevel(logging.ERROR)


class PdfProvider(BaseProvider):
    """
    A provider for PDF files.
    """

    page_range: Annotated[
        Optional[List[int]],
        "The range of pages to process.",
        "Default is None, which will process all pages.",
    ] = None
    pdftext_workers: Annotated[
        int,
        "The number of workers to use for pdftext.",
    ] = 4
    flatten_pdf: Annotated[
        bool,
        "Whether to flatten the PDF structure.",
    ] = True
    force_ocr: Annotated[
        bool,
        "Whether to force OCR on the whole document.",
    ] = False
    ocr_invalid_chars: Annotated[
        tuple,
        "The characters to consider invalid for OCR.",
    ] = (chr(0xFFFD), "�")
    ocr_space_threshold: Annotated[
        float,
        "The minimum ratio of spaces to non-spaces to detect bad text.",
    ] = 0.7
    ocr_newline_threshold: Annotated[
        float,
        "The minimum ratio of newlines to non-newlines to detect bad text.",
    ] = 0.6
    ocr_alphanum_threshold: Annotated[
        float,
        "The minimum ratio of alphanumeric characters to non-alphanumeric characters to consider an alphanumeric character.",
    ] = 0.3
    image_threshold: Annotated[
        float,
        "The minimum coverage ratio of the image to the page to consider skipping the page.",
    ] = 0.65
    strip_existing_ocr: Annotated[
        bool,
        "Whether to strip existing OCR text from the PDF.",
    ] = False
    disable_links: Annotated[
        bool,
        "Whether to disable links.",
    ] = False

    def __init__(self, filepath: str, config: Optional[BaseModel | dict] = None):
        super().__init__(filepath, config)

        self.filepath = filepath

        with self.get_doc() as doc:
            self.page_count = len(doc)
            self.page_lines: ProviderPageLines = {i: [] for i in range(len(doc))}
            self.page_refs: Dict[int, List[Reference]] = {
                i: [] for i in range(len(doc))
            }

            if self.page_range is None:
                self.page_range = list(range(len(doc)))

            if self.page_range:  # Check if page_range is not empty
                assert max(self.page_range) < len(doc) and min(self.page_range) >= 0, (
                    f"Invalid page range, values must be between 0 and {len(doc) - 1}.  Min of provided page range is {min(self.page_range)} and max is {max(self.page_range)}."
                )

            if self.force_ocr:
                # Manually assign page bboxes, since we can't get them from pdftext
                self.page_bboxes = {i: doc[i].get_bbox() for i in self.page_range}
            else:
                self.page_lines = self.pdftext_extraction(doc)

    @contextlib.contextmanager
    def get_doc(self):
        doc = None
        try:
            doc = pdfium.PdfDocument(self.filepath)

            # Must be called on the parent pdf, before retrieving pages to render correctly
            if self.flatten_pdf:
                doc.init_forms()

            yield doc
        finally:
            if doc:
                doc.close()

    def __len__(self) -> int:
        return self.page_count

    def font_flags_to_format(self, flags: Optional[int]) -> Set[str]:
        if flags is None:
            return {"plain"}

        flag_map = {
            1: "FixedPitch",
            2: "Serif",
            3: "Symbolic",
            4: "Script",
            6: "Nonsymbolic",
            7: "Italic",
            17: "AllCap",
            18: "SmallCap",
            19: "ForceBold",
            20: "UseExternAttr",
        }
        set_flags = set()
        for bit_position, flag_name in flag_map.items():
            if flags & (1 << (bit_position - 1)):
                set_flags.add(flag_name)
        if not set_flags:
            set_flags.add("Plain")

        formats = set()
        if set_flags == {"Symbolic", "Italic"} or set_flags == {
            "Symbolic",
            "Italic",
            "UseExternAttr",
        }:
            formats.add("plain")
        elif set_flags == {"UseExternAttr"}:
            formats.add("plain")
        elif set_flags == {"Plain"}:
            formats.add("plain")
        else:
            if set_flags & {"Italic"}:
                formats.add("italic")
            if set_flags & {"ForceBold"}:
                formats.add("bold")
            if set_flags & {
                "FixedPitch",
                "Serif",
                "Script",
                "Nonsymbolic",
                "AllCap",
                "SmallCap",
                "UseExternAttr",
            }:
                formats.add("plain")
        return formats

    def font_names_to_format(self, font_name: str | None) -> Set[str]:
        formats = set()
        if font_name is None:
            return formats

        if "bold" in font_name.lower():
            formats.add("bold")
        if "ital" in font_name.lower():
            formats.add("italic")
        return formats

    @staticmethod
    def normalize_spaces(text):
        space_chars = [
            "\u2003",  # em space
            "\u2002",  # en space
            "\u00a0",  # non-breaking space
            "\u200b",  # zero-width space
            "\u3000",  # ideographic space
        ]
        for space in space_chars:
            text = text.replace(space, " ")
        return text

    def pdftext_extraction(self, doc: PdfDocument) -> ProviderPageLines:
        page_lines: ProviderPageLines = {}
        page_char_blocks = dictionary_output(
            self.filepath,
            page_range=self.page_range,
            keep_chars=True,
            workers=self.pdftext_workers,
            flatten_pdf=self.flatten_pdf,
            quote_loosebox=False,
            disable_links=self.disable_links,
        )
        self.page_bboxes = {
            i: [0, 0, page["width"], page["height"]]
            for i, page in zip(self.page_range or [], page_char_blocks)
        }

        SpanClass = get_block_class(BlockTypes.Span)
        LineClass = get_block_class(BlockTypes.Line)
        CharClass = get_block_class(BlockTypes.Char)

        for page in page_char_blocks:
            page_id = page["page"]
            lines: List[ProviderOutput] = []
            if not self.check_page(page_id, doc):
                continue

            for block in page["blocks"]:
                for line in block["lines"]:
                    spans: List[Span] = []
                    chars: List[List[Char]] = []
                    for span in line["spans"]:
                        if not span["text"]:
                            continue
                        font_flags = span["font"]["flags"]
                        if isinstance(font_flags, str):
                            font_flags = None  # Convert string to None for type safety
                        font_formats = self.font_flags_to_format(
                            font_flags
                        ).union(self.font_names_to_format(span["font"]["name"]))
                        font_name = span["font"]["name"] or "Unknown"
                        font_weight = span["font"]["weight"] or 0
                        font_size = span["font"]["size"] or 0
                        polygon = PolygonBox.from_bbox(
                            span["bbox"], ensure_nonzero_area=True  # type: ignore
                        )
                        span_chars = [
                            CharClass(  # type: ignore
                                text=c["char"],  # type: ignore
                                idx=c["char_idx"],  # type: ignore
                                polygon=PolygonBox.from_bbox(
                                    c["bbox"], ensure_nonzero_area=True  # type: ignore
                                ),
                                page_id=page_id,
                                block_description="A single character inside a span.",
                            )
                            for c in span["chars"]
                        ]
                        superscript = span.get("superscript", False)
                        subscript = span.get("subscript", False)
                        text = self.normalize_spaces(fix_text(span["text"]))
                        if superscript or superscript:
                            text = text.strip()

                        spans.append(
                            SpanClass(  # type: ignore
                                polygon=polygon,
                                text=text,  # type: ignore
                                font=font_name,  # type: ignore
                                font_weight=font_weight,  # type: ignore
                                font_size=font_size,  # type: ignore
                                minimum_position=span["char_start_idx"],  # type: ignore
                                maximum_position=span["char_end_idx"],  # type: ignore
                                formats=list(font_formats),  # type: ignore
                                page_id=page_id,
                                text_extraction_method="pdftext",  # type: ignore
                                url=span.get("url"),  # type: ignore
                                has_superscript=superscript,  # type: ignore
                                has_subscript=subscript,  # type: ignore
                                block_description="A span of text inside a line.",
                            )
                        )
                        chars.append(span_chars)  # type: ignore
                    polygon = PolygonBox.from_bbox(
                        line["bbox"], ensure_nonzero_area=True  # type: ignore
                    )
                    assert len(spans) == len(chars)
                    lines.append(
                        ProviderOutput(
                            line=LineClass(polygon=polygon, page_id=page_id),  # type: ignore
                            spans=spans,
                            chars=chars,
                        )
                    )
            if self.check_line_spans(lines):
                page_lines[page_id] = lines

            self.page_refs[page_id] = []
            if page_refs := page.get("refs", None):
                self.page_refs[page_id] = page_refs  # type: ignore

        return page_lines

    def check_line_spans(self, page_lines: List[ProviderOutput]) -> bool:
        page_spans = [span for line in page_lines for span in line.spans]
        if len(page_spans) == 0:
            return False

        text = ""
        for span in page_spans:
            text = text + " " + span.text
            text = text + "\n"
        if len(text.strip()) == 0:
            return False
        if self.detect_bad_ocr(text):
            return False
        return True

    def check_page(self, page_id: int, doc: PdfDocument) -> bool:
        page = doc.get_page(page_id)
        page_bbox = PolygonBox.from_bbox(page.get_bbox())  # type: ignore
        try:
            page_objs = list(
                page.get_objects(
                    filter=[pdfium_c.FPDF_PAGEOBJ_TEXT, pdfium_c.FPDF_PAGEOBJ_IMAGE]
                )
            )
        except PdfiumError:
            # Happens when pdfium fails to get the number of page objects
            return False

        # if we do not see any text objects in the pdf, we can skip this page
        if not any([obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT for obj in page_objs]):  # type: ignore
            return False

        if self.strip_existing_ocr:
            # If any text objects on the page are in invisible render mode, skip this page
            for text_obj in filter(
                lambda obj: obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT, page_objs  # type: ignore
            ):
                if pdfium_c.FPDFTextObj_GetTextRenderMode(text_obj) in [
                    pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE,
                    pdfium_c.FPDF_TEXTRENDERMODE_UNKNOWN,
                ]:
                    return False

            non_embedded_fonts = []
            empty_fonts = []
            font_map = {}
            for text_obj in filter(
                lambda obj: obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT, page_objs  # type: ignore
            ):
                font = pdfium_c.FPDFTextObj_GetFont(text_obj)
                font_name = self._get_fontname(font)

                # we also skip pages without embedded fonts and fonts without names
                non_embedded_fonts.append(pdfium_c.FPDFFont_GetIsEmbedded(font) == 0)
                empty_fonts.append(
                    "glyphless" in font_name.lower()
                )  # Add font name check back in when we bump pypdfium2
                if font_name not in font_map:
                    font_map[font_name or "Unknown"] = font

            if all(non_embedded_fonts) or all(empty_fonts):
                return False

            # if we see very large images covering most of the page, we can skip this page
            for img_obj in filter(
                lambda obj: obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE, page_objs  # type: ignore
            ):
                img_bbox = PolygonBox.from_bbox(img_obj.get_pos())  # type: ignore
                if page_bbox.intersection_pct(img_bbox) >= self.image_threshold:
                    return False

        return True

    def detect_bad_ocr(self, text):
        if len(text) == 0:
            # Assume OCR failed if we have no text
            return True

        spaces = len(re.findall(r"\s+", text))
        alpha_chars = len(re.sub(r"\s+", "", text))
        if spaces / (alpha_chars + spaces) > self.ocr_space_threshold:
            return True

        newlines = len(re.findall(r"\n+", text))
        non_newlines = len(re.sub(r"\n+", "", text))
        if newlines / (newlines + non_newlines) > self.ocr_newline_threshold:
            return True

        if alphanum_ratio(text) < self.ocr_alphanum_threshold:  # Garbled text
            return True

        invalid_chars = len([c for c in text if c in self.ocr_invalid_chars])
        if invalid_chars > max(6.0, len(text) * 0.03):
            return True

        return False

    @staticmethod
    def _render_image(
        pdf: pdfium.PdfDocument, idx: int, dpi: int, flatten_page: bool
    ) -> Image.Image:
        page = pdf[idx]
        if flatten_page:
            flatten_pdf_page(page)
            page = pdf[idx]
        image = page.render(scale=int(dpi / 72), draw_annots=False).to_pil()
        image = image.convert("RGB")
        return image

    def get_images(self, idxs: List[int], dpi: int) -> List[Image.Image]:
        with self.get_doc() as doc:
            images = [
                self._render_image(doc, idx, dpi, self.flatten_pdf) for idx in idxs
            ]
        return images

    def get_page_bbox(self, idx: int) -> PolygonBox | None:
        bbox = self.page_bboxes.get(idx)
        if bbox:
            return PolygonBox.from_bbox(bbox)  # type: ignore

    def get_page_lines(self, idx: int) -> List[ProviderOutput]:
        return self.page_lines[idx]

    def get_page_refs(self, idx: int) -> List[Reference]:
        return self.page_refs[idx]

    @staticmethod
    def _get_fontname(font) -> str:
        font_name = ""
        buffer_size = 256

        try:
            font_name_buffer = ctypes.create_string_buffer(buffer_size)
            length = pdfium_c.FPDFFont_GetBaseFontName(  # type: ignore
                font, font_name_buffer, buffer_size
            )
            if length < buffer_size:
                font_name = font_name_buffer.value.decode("utf-8")
            else:
                font_name_buffer = ctypes.create_string_buffer(length)
                pdfium_c.FPDFFont_GetBaseFontName(font, font_name_buffer, length)  # type: ignore
                font_name = font_name_buffer.value.decode("utf-8")
        except Exception:
            pass

        return font_name
