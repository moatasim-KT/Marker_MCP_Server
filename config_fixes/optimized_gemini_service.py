import json
import time
from io import BytesIO
from typing import List, Annotated

import PIL
from google import genai
from google.genai import types
from google.genai.errors import APIError
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services.gemini import BaseGeminiService

logger = get_logger()


class OptimizedGeminiService(BaseGeminiService):
    """
    Optimized Gemini service with better payload management and error handling
    """
    
    max_image_size: Annotated[int, "Maximum image size in pixels"] = 1024 * 1024  # 1MP
    max_prompt_length: Annotated[int, "Maximum prompt length in characters"] = 20000
    compression_quality: Annotated[int, "JPEG compression quality for large images"] = 80
    
    def compress_image_if_needed(self, img: PIL.Image.Image) -> PIL.Image.Image:
        """Compress image if it's too large to reduce payload size"""
        width, height = img.size
        total_pixels = width * height
        
        if total_pixels > self.max_image_size:
            # Calculate new dimensions
            scale_factor = (self.max_image_size / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize image
            img = img.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
            logger.info(f"Compressed image from {width}x{height} to {new_width}x{new_height}")
        
        return img

    def img_to_bytes(self, img: PIL.Image.Image):
        # Compress image if needed
        img = self.compress_image_if_needed(img)
        
        image_bytes = BytesIO()
        # Use JPEG with compression for smaller payload
        img.save(image_bytes, format="JPEG", quality=self.compression_quality, optimize=True)
        return image_bytes.getvalue()

    def truncate_prompt_if_needed(self, prompt: str) -> str:
        """Truncate prompt if it's too long"""
        if len(prompt) > self.max_prompt_length:
            # Keep the beginning and end, truncate middle
            truncate_msg = f"\n\n[... content truncated to stay within {self.max_prompt_length} characters ...]\n\n"
            half_length = (self.max_prompt_length - len(truncate_msg)) // 2
            truncated = prompt[:half_length] + truncate_msg + prompt[-half_length:]
            logger.warning(f"Prompt truncated from {len(prompt)} to {len(truncated)} characters")
            return truncated
        return prompt

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

        # Truncate prompt if too long
        prompt = self.truncate_prompt_if_needed(prompt)

        client = self.get_google_client(timeout=timeout)
        
        try:
            image_parts = [
                types.Part.from_bytes(data=self.img_to_bytes(img), mime_type="image/jpeg")
                for img in image
            ]
        except Exception as e:
            logger.error(f"Error processing images: {e}")
            block.update_metadata(llm_error_count=1)
            return {}

        tries = 0
        while tries < max_retries:
            try:
                responses = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=image_parts + [prompt],
                    config={
                        "temperature": 0,
                        "response_schema": response_schema,
                        "response_mime_type": "application/json",
                    },
                )
                
                if not responses.candidates or not responses.candidates[0].content.parts:
                    logger.warning("Empty response from Gemini")
                    block.update_metadata(llm_error_count=1)
                    return {}
                
                output = responses.candidates[0].content.parts[0].text
                total_tokens = responses.usage_metadata.total_token_count
                block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                
                # Validate JSON before returning
                try:
                    parsed_output = json.loads(output)
                    return parsed_output
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON response: {e}")
                    block.update_metadata(llm_error_count=1)
                    return {}
                    
            except APIError as e:
                if e.code in [429, 443, 503]:
                    # Rate limit exceeded - use exponential backoff
                    tries += 1
                    wait_time = min(tries * self.retry_wait_time * 2, 60)  # Cap at 60 seconds
                    logger.warning(
                        f"Rate limit exceeded (code {e.code}). Retrying in {wait_time} seconds... (Attempt {tries}/{max_retries})"
                    )
                    time.sleep(wait_time)
                elif e.code == 400:
                    # Bad request - likely payload too large
                    logger.error(f"Bad request (code 400): {e}. Payload may be too large.")
                    block.update_metadata(llm_error_count=1)
                    break
                else:
                    logger.error(f"APIError: {e}")
                    block.update_metadata(llm_error_count=1)
                    break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                block.update_metadata(llm_error_count=1)
                break

        return {}