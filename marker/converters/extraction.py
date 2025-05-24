import json
import re

from typing import Union, cast, Any
from marker.builders.document import DocumentBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.converters.pdf import PdfConverter
from marker.extractors.page import PageExtractor, json_schema_to_base_model
from marker.providers import BaseProvider
from marker.providers.registry import provider_from_filepath

from marker.renderers.extraction import ExtractionRenderer, ExtractionOutput
from marker.renderers.markdown import MarkdownRenderer

from marker.logger import get_logger

logger = get_logger()


class ExtractionConverter(PdfConverter):
    pattern: str = r"{\d+\}-{48}\n\n"

    def _get_config_value(self, key: str):
        """Helper method to safely access config values regardless of type."""
        if self.config is None:
            return None
        if isinstance(self.config, dict):
            return self.config.get(key)
        else:
            return getattr(self.config, key, None)
    
    def _set_config_value(self, key: str, value):
        """Helper method to safely set config values regardless of type."""
        if self.config is None:
            self.config = {}
        if isinstance(self.config, dict):
            self.config[key] = value
        else:
            setattr(self.config, key, value)

    def build_document(self, filepath: str):
        provider_cls = provider_from_filepath(filepath)
        layout_builder = self.resolve_dependencies(self.layout_builder_class)
        line_builder = self.resolve_dependencies(LineBuilder)
        ocr_builder = self.resolve_dependencies(OcrBuilder)
        provider = provider_cls(filepath, self.config)
        
        # Cast to expected type since DocumentBuilder expects PdfProvider specifically
        # but this converter handles multiple provider types
        document = DocumentBuilder(self.config)(
            cast(Any, provider), layout_builder, line_builder, ocr_builder
        )
        structure_builder_cls = self.resolve_dependencies(StructureBuilder)
        structure_builder_cls(document)

        for processor in self.processor_list:
            processor(document)

        return document, provider

    def __call__(self, filepath: str) -> ExtractionOutput:
        self._set_config_value("paginate_output", True)  # Ensure we can split the output properly
        self._set_config_value("output_format", "markdown")  # Output must be markdown for extraction
        
        try:
            page_schema = self._get_config_value("page_schema")
            if page_schema:
                json_schema_to_base_model(json.loads(page_schema))
        except Exception as e:
            logger.error(f"Could not parse page schema: {e}")
            raise ValueError(
                "Could not parse your page schema. Please check the schema format."
            )

        document, provider = self.build_document(filepath)
        renderer = self.resolve_dependencies(MarkdownRenderer)
        output = renderer(document)

        output_pages = re.split(self.pattern, output.markdown)[
            1:
        ]  # Split output into pages

        # This needs an LLM service for extraction, this sets it in the extractor
        if not self.artifact_dict["llm_service"]:
            self.artifact_dict["llm_service"] = self.resolve_dependencies(
                self.default_llm_service
            )

        extractor = self.resolve_dependencies(PageExtractor)
        renderer = self.resolve_dependencies(ExtractionRenderer)

        pnums = provider.page_range or []  # Handle None case
        all_json = {}
        for page, page_md, pnum in zip(document.pages, output_pages, pnums):
            extracted_json = extractor(document, page, page_md.strip())
            all_json[pnum] = extracted_json

        merged = renderer(all_json)
        return merged
