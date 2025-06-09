import base64
import json
import time
from io import BytesIO
from typing import Annotated, List, Union, Optional

import openai
import PIL.Image
from marker.logger import get_logger
from openai import APITimeoutError, RateLimitError
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class NvidiaService(BaseService):
    nvidia_base_url: Annotated[
        str, "The base url to use for NVIDIA models.  No trailing slash."
    ] = "https://integrate.api.nvidia.com/v1"
    nvidia_model: Annotated[str, "The model name to use for NVIDIA model."] = (
        "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    )
    nvidia_api_key: Annotated[
        Optional[str], "The API key to use for the NVIDIA service."
    ] = None

    def image_to_base64(self, image: PIL.Image.Image):
        image_bytes = BytesIO()
        image.save(image_bytes, format="WEBP")
        return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    def prepare_images(
        self, images: Union[PIL.Image.Image, List[PIL.Image.Image]]
    ) -> List[dict]:
        if isinstance(images, PIL.Image.Image):
            images = [images]

        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/webp;base64,{}".format(
                        self.image_to_base64(img)
                    ),
                },
            }
            for img in images
        ]

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image],
        block: Block,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        if not isinstance(image, list):
            image = [image]

        client = self.get_client()
        image_data = self.prepare_images(image)

        messages = [  # type: ignore
            {
                "role": "user",
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        tries = 0
        while tries < max_retries:
            try:
                # NVIDIA API uses streaming by default, but we need to handle it properly
                # For structured output, we'll use non-streaming mode if available
                # or collect the streaming response
                response = client.chat.completions.create(
                    model=self.nvidia_model,
                    messages=messages,  # type: ignore
                    timeout=timeout,
                    temperature=1.00,
                    top_p=0.01,
                    max_tokens=1024,
                    stream=False,  # Disable streaming for structured output
                )
                
                response_text = response.choices[0].message.content
                if response.usage and response.usage.total_tokens:
                    total_tokens = response.usage.total_tokens
                    block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                
                if response_text:
                    # Try to parse as JSON for structured output
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        # If not valid JSON, try to extract JSON from the response
                        # This is common with instruction-following models
                        import re
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            try:
                                return json.loads(json_match.group())
                            except json.JSONDecodeError:
                                pass
                        
                        # If we can't parse JSON, log the response and return empty dict
                        logger.warning(f"Could not parse JSON from NVIDIA response: {response_text}")
                        return {}
                return {}
            except (APITimeoutError, RateLimitError) as e:
                # Rate limit exceeded
                tries += 1
                wait_time = tries * self.retry_wait_time
                logger.warning(
                    f"Rate limit error: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{max_retries})"
                )
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"NVIDIA inference failed: {e}")
                break

        return {}

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.nvidia_api_key, base_url=self.nvidia_base_url)
