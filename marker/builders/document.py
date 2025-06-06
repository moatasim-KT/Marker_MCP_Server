from typing import Annotated, Type, cast

from marker.builders import BaseBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.providers import BaseProvider
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.registry import get_block_class


class DocumentBuilder(BaseBuilder):
    """
    Constructs a Document given a PdfProvider, LayoutBuilder, and OcrBuilder.
    """
    lowres_image_dpi: Annotated[
        int,
        "DPI setting for low-resolution page images used for Layout and Line Detection.",
    ] = 96
    highres_image_dpi: Annotated[
        int,
        "DPI setting for high-resolution page images used for OCR.",
    ] = 192
    disable_ocr: Annotated[
        bool,
        "Disable OCR processing.",
    ] = False

    def __call__(self, provider: BaseProvider, layout_builder: LayoutBuilder, line_builder: LineBuilder, ocr_builder: OcrBuilder):
        document = self.build_document(provider)
        # Cast provider to PdfProvider for builder compatibility
        pdf_provider = cast(PdfProvider, provider)
        layout_builder(document, pdf_provider)
        line_builder(document, pdf_provider)
        if not self.disable_ocr:
            ocr_builder(document, pdf_provider)
        return document

    def build_document(self, provider: BaseProvider):
        PageGroupClass: Type[PageGroup] = cast(Type[PageGroup], get_block_class(BlockTypes.Page))
        
        # Get page_range from provider, default to all pages if not available
        page_range = getattr(provider, 'page_range', None)
        if page_range is None:
            # Handle the case where provider doesn't have a proper length implementation
            # Try to get page_count directly if available (e.g., PdfProvider)
            provider_length = getattr(provider, 'page_count', None)
            if provider_length is not None:
                page_range = list(range(provider_length))
            else:
                # Try calling __len__ directly and handle None return
                try:
                    provider_len_method = getattr(provider, '__len__', None)
                    if provider_len_method is not None:
                        provider_len = provider_len_method()
                        if provider_len is not None and isinstance(provider_len, int):
                            page_range = list(range(provider_len))
                        else:
                            # __len__ returned None or non-integer, fallback
                            page_range = [0]
                    else:
                        # No __len__ method, fallback
                        page_range = [0]
                except (TypeError, AttributeError):
                    # Fallback for any other errors
                    page_range = [0]
        
        lowres_images = provider.get_images(page_range, self.lowres_image_dpi)
        highres_images = provider.get_images(page_range, self.highres_image_dpi)
        initial_pages = []
        for i, p in enumerate(page_range):
            page_bbox = provider.get_page_bbox(p)
            if page_bbox is not None:
                page = PageGroupClass(
                    page_id=p,
                    lowres_image=lowres_images[i],
                    highres_image=highres_images[i],
                    polygon=page_bbox,
                    refs=provider.get_page_refs(p)
                )
                initial_pages.append(page)
        DocumentClass: Type[Document] = cast(Type[Document], get_block_class(BlockTypes.Document))
        return DocumentClass(filepath=provider.filepath, pages=initial_pages)
