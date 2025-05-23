import os
import pytest
from marker.services.groq import GroqService
from marker.schema.blocks import Block
from pydantic import BaseModel
from PIL import Image

class DummySchema(BaseModel):
    answer: str

def test_groq_service_call(monkeypatch):
    # Only run if GROQ_API_KEY is set
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")
    service = GroqService({
        'groq_model_name': 'compound-beta',
        'groq_api_key': api_key,
    })
    # Dummy block and prompt
    from marker.schema.polygon import PolygonBox
    from marker.schema import BlockTypes
    # PolygonBox expects a 'polygon' argument: 4 corners, each [x, y]
    block = Block(
        polygon=PolygonBox(polygon=[[0,0],[1,0],[1,1],[0,1]]),
        block_description="Test block",
        block_type=BlockTypes.Text,
        page_id=1
    )
    prompt = "Return a JSON object with answer: 'hello world'"
    # No image for this test
    result = service(prompt, [], block, DummySchema)
    assert isinstance(result, dict)
    assert 'answer' in result
