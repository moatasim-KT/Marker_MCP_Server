from typing import Tuple

from marker.builders.document import DocumentBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.converters.pdf import PdfConverter
from marker.processors import BaseProcessor
from marker.processors.equation import EquationProcessor
from marker.providers.registry import provider_from_filepath
from marker.renderers.ocr_json import OCRJSONRenderer
from marker.models import create_model_dict


class OCRConverter(PdfConverter):
    default_processors: Tuple[type, ...] = (EquationProcessor,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.config:
            self.config = {}

        self.config["format_lines"] = True
        self.renderer = OCRJSONRenderer

    def initialize_processors(self, processor_classes):
        # Custom instantiation for EquationProcessor
        processors = []
        model_dict = create_model_dict()
        for proc in processor_classes:
            if proc is EquationProcessor:
                processors.append(EquationProcessor(model_dict["recognition_model"]))
            else:
                processors.append(proc())
        return processors

    def build_document(self, filepath: str):
        provider_cls = provider_from_filepath(filepath)
        # Ensure provider is a PdfProvider for DocumentBuilder
        from marker.providers.pdf import PdfProvider
        if provider_cls is not PdfProvider:
            raise TypeError(f"OCRConverter only supports PdfProvider, got {provider_cls.__name__}")
        layout_builder = self.resolve_dependencies(self.layout_builder_class)
        line_builder = self.resolve_dependencies(LineBuilder)
        ocr_builder = self.resolve_dependencies(OcrBuilder)
        document_builder = DocumentBuilder(self.config)

        provider = provider_cls(filepath, self.config)
        document = document_builder(provider, layout_builder, line_builder, ocr_builder)

        if self.processor_list is not None:
            for processor in self.processor_list:
                if callable(processor):
                    processor(document)

        return document

    def __call__(self, filepath: str):
        document = self.build_document(filepath)
        renderer_cls = self.resolve_dependencies(self.renderer)
        renderer = renderer_cls if callable(renderer_cls) else OCRJSONRenderer
        return renderer(document)
