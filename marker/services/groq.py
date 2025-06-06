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


class GroqService(BaseService):
    groq_base_url: Annotated[
        str, "The base url to use for Groq models.  No trailing slash."
    ] = "https://api.groq.com/openai/v1"
    groq_model: Annotated[str, "The model name to use for Groq."] = (
        "llama-3.3-70b-versatile"
    )
    groq_api_key: Annotated[
        Optional[str], "The API key to use for the Groq service."
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

        tries = 0
        while tries < max_retries:
            try:
                # Groq doesn't support json_schema response format, so we use regular chat completion
                # and add JSON formatting instructions to the prompt
                json_prompt = f"""{prompt}

Please respond with valid JSON only, following this exact schema:
{response_schema.model_json_schema()}

Ensure your response is valid JSON that matches the schema exactly."""

                # Check if Groq model supports vision
                # Groq vision models: llama-3.2-11b-vision-preview, llama-3.2-90b-vision-preview
                is_vision_model = "vision" in self.groq_model.lower()
                
                if is_vision_model and len(image_data) > 0:
                    # Use vision format for models that support it
                    enhanced_messages = [  # type: ignore
                        {
                            "role": "user",
                            "content": [
                                *image_data,
                                {"type": "text", "text": json_prompt},
                            ],
                        }
                    ]
                else:
                    # For non-vision models, use text-only format and warn about image loss
                    if len(image_data) > 0 and not is_vision_model:
                        logger.warning(f"Model {self.groq_model} does not support vision. Image data will be ignored.")
                    
                    enhanced_messages = [  # type: ignore
                        {
                            "role": "user",
                            "content": json_prompt,
                        }
                    ]

                response = client.chat.completions.create(
                    extra_headers={
                        "X-Title": "Marker",
                        "HTTP-Referer": "https://github.com/VikParuchuri/marker",
                    },
                    model=self.groq_model,
                    messages=enhanced_messages,  # type: ignore
                    timeout=timeout,
                    temperature=0,  # Use deterministic output for better JSON parsing
                )
                response_text = response.choices[0].message.content
                if response.usage and response.usage.total_tokens:
                    total_tokens = response.usage.total_tokens
                    block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                
                if response_text:
                    # Clean up the response text to extract JSON
                    cleaned_text = response_text.strip()
                    
                    # Try to find JSON content if it's wrapped in markdown or other text
                    if "```json" in cleaned_text:
                        start = cleaned_text.find("```json") + 7
                        end = cleaned_text.find("```", start)
                        if end != -1:
                            cleaned_text = cleaned_text[start:end].strip()
                    elif "```" in cleaned_text:
                        start = cleaned_text.find("```") + 3
                        end = cleaned_text.find("```", start)
                        if end != -1:
                            cleaned_text = cleaned_text[start:end].strip()
                    
                    try:
                        return json.loads(cleaned_text)
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"Failed to parse JSON response from Groq: {json_err}")
                        logger.debug(f"Raw response: {response_text}")
                        # Try to extract JSON from the middle of the response
                        import re
                        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                        if json_match:
                            try:
                                return json.loads(json_match.group())
                            except json.JSONDecodeError:
                                pass
                        # If all else fails, return empty dict
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
                logger.error(f"Groq inference failed: {e}")
                break

        return {}

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.groq_api_key, base_url=self.groq_base_url)
