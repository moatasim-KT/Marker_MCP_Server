import contextlib
import ctypes
import logging
import re
from typing import Annotated, Dict, List, Optional, Set, Generator
import typing

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from ftfy import fix_text
from pdftext.extraction import dictionary_output
from pdftext.schema import Reference
from pdftext.pdf.utils import flatten as flatten_pdf_page

from PIL import Image
from pypdfium2 import PdfiumError, PdfDocument

from marker.providers import BaseProvider, ProviderOutput, Char, ProviderPageLines
from marker.providers.utils import alphanum_ratio
from marker.schema import BlockTypes
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class
from marker.schema.text.line import Line
from marker.schema.text.span import Span
from marker.schema.text.char import Char

# DEBUG: Test direct instantiation of Char, Span, Line
from marker.schema.polygon import PolygonBox
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

    def __init__(self, filepath: str, config=None):
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
            elif isinstance(self.page_range, range):
                self.page_range = list(self.page_range)

            assert self.page_range is not None and len(self.page_range) > 0, "page_range must be a non-empty list of ints."
            assert max(self.page_range) < len(doc) and min(self.page_range) >= 0, (
                f"Invalid page range, values must be between 0 and {len(doc) - 1}.  Min of provided page range is {min(self.page_range)} and max is {max(self.page_range)}."
            )

            if self.force_ocr:
                # Manually assign page bboxes, since we can't get them from pdftext
                self.page_bboxes = {i: doc[i].get_bbox() for i in self.page_range}
            else:
                self.page_lines = self.pdftext_extraction(doc)

    @contextlib.contextmanager
    def get_doc(self) -> Generator[PdfDocument, None, None]:
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
            page_range=self.page_range if self.page_range is not None else [],
            keep_chars=True,
            workers=self.pdftext_workers,
            flatten_pdf=self.flatten_pdf,
            quote_loosebox=False,
            disable_links=self.disable_links,
        )
        self.page_bboxes = {
            i: [0.0, 0.0, float(page["width"]), float(page["height"])]
            for i, page in zip(self.page_range if self.page_range is not None else [], page_char_blocks)
        }

        SpanClass = Span
        LineClass = Line
        CharClass = Char

        for page in page_char_blocks:
            page_id = page["page"]
            lines: List[ProviderOutput] = []
            if not self.check_page(page_id, doc):
                continue

            for block in page["blocks"]:
                for line in block["lines"]:
                    spans: List = []
                    chars: List[List] = []
                    for span in line["spans"]:
                        if not span["text"]:
                            continue
                        font_flags = span["font"].get("flags")
                        if isinstance(font_flags, str):
                            try:
                                font_flags = int(font_flags)
                            except Exception:
                                font_flags = None
                        font_formats = list(self.font_flags_to_format(
                            font_flags
                        ).union(self.font_names_to_format(span["font"].get("name"))))
                        allowed_formats = {"plain", "math", "chemical", "bold", "italic", "highlight", "subscript", "superscript", "small", "code", "underline"}
                        font_formats = [f for f in font_formats if f in allowed_formats]
                        font_formats_literal = typing.cast(
                            list[typing.Literal[
                                "plain", "math", "chemical", "bold", "italic", "highlight", "subscript", "superscript", "small", "code", "underline"
                            ]],
                            font_formats,
                        )
                        font_name = span["font"].get("name") or "Unknown"
                        font_weight = float(span["font"].get("weight") or 0)
                        font_size = float(span["font"].get("size") or 0)
                        bbox = span["bbox"]
                        if hasattr(bbox, 'bbox'):
                            bbox = bbox.bbox
                        bbox = [float(x) for x in bbox]
                        polygon = PolygonBox.from_bbox(bbox, ensure_nonzero_area=True)
                        span_chars = [
                            CharClass(
                                polygon=PolygonBox.from_bbox([float(xx) for xx in (c["bbox"].bbox if hasattr(c["bbox"], 'bbox') else c["bbox"] )], ensure_nonzero_area=True),
                                text=c["char"],
                                idx=int(c["char_idx"]),
                                page_id=page_id,
                            )
                            for c in span["chars"]
                        ]
                        superscript = span.get("superscript", False)
                        subscript = span.get("subscript", False)
                        text = self.normalize_spaces(fix_text(span["text"]))
                        if superscript or subscript:
                            text = text.strip()

                        spans.append(
                            SpanClass(
                                polygon=polygon,
                                text=text,
                                font=font_name,
                                font_weight=font_weight,
                                font_size=font_size,
                                minimum_position=int(span["char_start_idx"]),
                                maximum_position=int(span["char_end_idx"]),
                                formats=font_formats_literal,
                                has_superscript=bool(superscript),
                                has_subscript=bool(subscript),
                                url=span.get("url"),
                                page_id=page_id,
                            )
                        )
                        chars.append(span_chars)
                    bbox = line["bbox"]
                    if hasattr(bbox, 'bbox'):
                        bbox = bbox.bbox
                    bbox = [float(x) for x in bbox]
                    polygon = PolygonBox.from_bbox(bbox, ensure_nonzero_area=True)
                    assert len(spans) == len(chars)
                    lines.append(
                        ProviderOutput(
                            line=LineClass(
                                polygon=polygon,
                                page_id=page_id,
                            ),
                            spans=spans,
                            chars=chars,
                        )
                    )
            if self.check_line_spans(lines):
                page_lines[page_id] = lines

            self.page_refs[page_id] = []
            if page_refs := page.get("refs", None):
                self.page_refs[page_id] = page_refs

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
        page_bbox = PolygonBox.from_bbox(list(page.get_bbox()))
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
        def get_obj_type(obj):
            return getattr(obj, 'type', None)

        if all(
            get_obj_type(obj) != pdfium_c.FPDF_PAGEOBJ_TEXT for obj in page_objs
        ):
            return False

        if self.strip_existing_ocr:
            # If any text objects on the page are in invisible render mode, skip this page
            for text_obj in filter(
                lambda obj: get_obj_type(obj) == pdfium_c.FPDF_PAGEOBJ_TEXT, page_objs
            ):
                try:
                    render_mode = pdfium_c.FPDFTextObj_GetTextRenderMode(text_obj)
                except Exception:
                    continue
                if render_mode in [
                    getattr(pdfium_c, 'FPDF_TEXTRENDERMODE_INVISIBLE', -1),
                    getattr(pdfium_c, 'FPDF_TEXTRENDERMODE_UNKNOWN', -1),
                ]:
                    return False

            non_embedded_fonts = []
            empty_fonts = []
            font_map = {}
            for text_obj in filter(
                lambda obj: get_obj_type(obj) == pdfium_c.FPDF_PAGEOBJ_TEXT, page_objs
            ):
                font = pdfium_c.FPDFTextObj_GetFont(text_obj)
                font_name = self._get_fontname(font)

                # we also skip pages without embedded fonts and fonts without names
                try:
                    is_embedded = pdfium_c.FPDFFont_GetIsEmbedded(font) == 0
                except Exception:
                    is_embedded = True
                non_embedded_fonts.append(is_embedded)
                empty_fonts.append(
                    "glyphless" in font_name.lower()
                )  # Add font name check back in when we bump pypdfium2
                if font_name not in font_map:
                    font_map[font_name or "Unknown"] = font

            if all(non_embedded_fonts) or all(empty_fonts):
                return False

            # if we see very large images covering most of the page, we can skip this page
            for img_obj in filter(
                lambda obj: get_obj_type(obj) == pdfium_c.FPDF_PAGEOBJ_IMAGE, page_objs
            ):
                img_bbox = PolygonBox.from_bbox(list(img_obj.get_pos()))
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
        # pdfium expects int for scale, so round to nearest int
        scale = int(round(dpi / 72))
        image = page.render(scale=scale, draw_annots=False).to_pil()
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
            bbox = [float(x) for x in bbox]
            return PolygonBox.from_bbox(bbox)

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
            # Fallback for missing FPDFFont_GetBaseFontName
            get_base_font_name = getattr(pdfium_c, 'FPDFFont_GetBaseFontName', None)
            if get_base_font_name is None:
                return "Unknown"
            length = get_base_font_name(
                font, font_name_buffer, buffer_size
            )
            if length < buffer_size:
                font_name = font_name_buffer.value.decode("utf-8")
            else:
                font_name_buffer = ctypes.create_string_buffer(length)
                get_base_font_name(font, font_name_buffer, length)
                font_name = font_name_buffer.value.decode("utf-8")
        except Exception:
            pass

        return font_name
