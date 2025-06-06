import base64
import io
import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from typing import Annotated, Optional

import click
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings

app_data = {}


UPLOAD_DIRECTORY = os.path.join(tempfile.gettempdir(), "marker_uploads")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


class CommonParams(BaseModel):
    filepath: Annotated[
        str | None, Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        str | None,
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20"
        )
    ] = None
    force_ocr: Annotated[
        bool,
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases)."
        ),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines)."
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(
            description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'."
        ),
    ] = "markdown"


async def _convert_pdf(params: CommonParams):
    assert params.output_format in ["markdown", "json", "html"], "Invalid output format"
    try:
        options = params.model_dump(exclude_none=True)
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter_cls = PdfConverter
        filepath_arg = params.filepath
        if filepath_arg is None:
            raise ValueError("Filepath must be provided for conversion.")
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(filepath_arg)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(
            settings.OUTPUT_ENCODING
        )

    return {
        "format": params.output_format,
        "output": text,
        "images": encoded,
        "metadata": metadata,
        "success": True,
    }


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    return await _convert_pdf(params)


@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: str | None = Form(default=None),
    force_ocr: bool = Form(default=False),
    paginate_output: bool = Form(default=False),
    output_format: str = Form(default="markdown"),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    if file.filename is None:
        raise ValueError("Uploaded file must have a filename.")
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    params = CommonParams(
        filepath=upload_path,
        page_range=page_range,
        force_ocr=force_ocr,
        paginate_output=paginate_output,
        output_format=output_format,
    )
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    import uvicorn

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
    )
