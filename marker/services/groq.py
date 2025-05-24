import base64
import json
import os
import re
import time
from io import BytesIO
from typing import List, Annotated, Union

import PIL
from PIL import Image
import requests
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class GroqService(BaseService):
    groq_api_key: Annotated[str, "The Groq API key to use for the service."] = ""
    groq_model_name: Annotated[str, "The Groq model name to use."] = "compound-beta"
    groq_base_url: Annotated[str, "The base url for Groq API."] = "https://api.groq.com/openai/v1"
    max_groq_tokens: Annotated[int, "The maximum number of tokens to use for a single Groq request."] = 8192
    # Prioritized list of models to cycle through on rate limit
    groq_model_list: List[str] = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen-qwq-32b",
        "deepseek-r1-distill-llama-70b",
        "llama-3.3-70b-versatile",
        "compound-beta",
        "compound-beta-mini",
    ]

    def __init__(self, config=None):
        # Always prefer environment variables if set
        api_key = os.environ.get("GROQ_API_KEY")
        base_url = os.environ.get("GROQ_BASE_URL")
        config_dict = dict(config) if config else {}
        # Override groq_api_key and groq_base_url from env if not provided
        if not config_dict.get('groq_api_key') and api_key:
            config_dict['groq_api_key'] = api_key
        if not config_dict.get('groq_base_url') and base_url:
            config_dict['groq_base_url'] = base_url
        self._model_index = 0
        if 'groq_model_name' in config_dict:
            # Start at the specified model if present in the list
            try:
                self._model_index = self.groq_model_list.index(config_dict['groq_model_name'])
            except ValueError:
                self._model_index = 0
        self.groq_model_name = self.groq_model_list[self._model_index]
        super().__init__(config_dict)

    def image_to_base64(self, image: Image.Image):
        image_bytes = BytesIO()
        image.save(image_bytes, format="WEBP")
        return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    def prepare_images(self, images: Union[Image.Image, List[Image.Image]]) -> List[dict]:
        if isinstance(images, Image.Image):
            images = [images]
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/webp;base64,{}".format(self.image_to_base64(img)),
                },
            }
            for img in images
        ]

    def _send_groq_request(self, payload, headers, timeout):
        try:
            response = requests.post(
                f"{self.groq_base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"[GroqService] HTTPError: {e}")
            print(f"[GroqService] Response content: {getattr(e.response, 'text', '')}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request to Groq API failed: {e}")
        return None

    def __call__(
        self,
        prompt: str,
        image: Image.Image | List[Image.Image],
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
        image_data = self.prepare_images(image)
        json_content = json.dumps([
            *image_data,
            {"type": "text", "text": prompt},
        ])
        messages = [
            {
                "role": "user",
                "content": json_content,
            }
        ]
        tries = 0
        model_attempts = 0
        max_model_attempts = len(self.groq_model_list)
        while tries < max_retries and model_attempts < max_model_attempts:
            try:
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.groq_model_name,
                    "messages": messages,
                    "max_tokens": self.max_groq_tokens,
                }
                print(f"[GroqService] Using model: {self.groq_model_name}")
                print(f"[GroqService] Payload model: {payload['model']}")
                print(f"[GroqService] Full payload: {json.dumps(payload, indent=2)}")
                try:
                    response = requests.post(
                        f"{self.groq_base_url}/chat/completions",
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=timeout,
                    )
                    if response.status_code == 429:
                        # Rate limit: try next model
                        model_attempts += 1
                        self._model_index = (self._model_index + 1) % len(self.groq_model_list)
                        self.groq_model_name = self.groq_model_list[self._model_index]
                        logger.warning(f"Groq rate limit hit. Switching to next model: {self.groq_model_name}")
                        time.sleep(2 * model_attempts)
                        continue
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    print(f"[GroqService] HTTPError: {e}")
                    print(f"[GroqService] Response content: {getattr(e.response, 'text', '')}")
                    raise
                result = response.json()
                response_text = result["choices"][0]["message"]["content"] if result.get("choices") else ""
                print(f"[GroqService] Raw API response: {result}")
                print(f"[GroqService] Response text: {response_text}")
                total_tokens = result.get("usage", {}).get("total_tokens", 0)
                block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                if not response_text:
                    logger.error("Groq API returned empty response text.")
                    return {}
                # Try to extract JSON from the response text
                json_str = None
                # Look for ```json ... ``` block
                match = re.search(r"```json\s*([\s\S]+?)```", response_text)
                if match:
                    json_str = match.group(1).strip()
                else:
                    # Fallback: look for first {...} block
                    match = re.search(r"\{[\s\S]+\}", response_text)
                    if match:
                        json_str = match.group(0)
                if not json_str:
                    logger.error(f"Groq response does not contain JSON. Raw: {response_text}")
                    return {}
                try:
                    return json.loads(json_str)
                except Exception as e:
                    logger.error(f"Groq extracted JSON is not valid: {e}\nRaw: {json_str}")
                    return {}
            except requests.exceptions.RequestException as e:
                tries += 1
                wait_time = tries * self.retry_wait_time
                logger.warning(
                    f"Groq API error: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{max_retries})"
                )
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Groq inference failed: {e}")
                break
        # Chunking logic: if payload is too large, split and send concurrently
        import concurrent.futures
        max_payload_tokens = self.max_groq_tokens
        est_tokens = len(json_content) // 4
        if est_tokens > max_payload_tokens:
            chunk_size = max_payload_tokens * 4  # chars
            chunks = [json_content[i:i+chunk_size] for i in range(0, len(json_content), chunk_size)]
            results = []
            failed_chunks = []
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            }
            timeout = timeout or self.timeout
            def process_chunk(chunk, model_name):
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": chunk}],
                    "max_tokens": self.max_groq_tokens,
                }
                result = self._send_groq_request(payload, headers, timeout)
                if not result or not result.get("choices"):
                    logger.warning(f"Chunk failed for model {model_name}. Retrying with next model.")
                    # Try all models for this chunk
                    for alt_model in self.groq_model_list:
                        if alt_model == model_name:
                            continue
                        payload["model"] = alt_model
                        result = self._send_groq_request(payload, headers, timeout)
                        if result and result.get("choices"):
                            logger.info(f"Chunk succeeded with fallback model {alt_model}.")
                            break
                    else:
                        logger.error(f"All models failed for chunk. Skipping chunk.")
                        return None
                # Extract JSON from response
                response_text = result["choices"][0]["message"]["content"]
                json_str = None
                match = re.search(r"```json\s*([\s\S]+?)```", response_text)
                if match:
                    json_str = match.group(1).strip()
                else:
                    match = re.search(r"\{[\s\S]+\}", response_text)
                    if match:
                        json_str = match.group(0)
                if not json_str:
                    logger.error(f"Groq response does not contain JSON. Raw: {response_text}")
                    return None
                try:
                    return json.loads(json_str)
                except Exception as e:
                    logger.error(f"Groq extracted JSON is not valid: {e}\nRaw: {json_str}")
                    return None
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.groq_model_list)) as executor:
                futures = []
                for i, chunk in enumerate(chunks):
                    model_name = self.groq_model_list[i % len(self.groq_model_list)]
                    futures.append(executor.submit(process_chunk, chunk, model_name))
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        failed_chunks.append(i)
            # Aggregate valid JSON results only
            if not results:
                logger.error("All chunks failed. Returning empty result.")
                return {}
            # Simple aggregation: merge dicts if possible, else return list
            if all(isinstance(r, dict) for r in results):
                aggregated = {}
                for r in results:
                    aggregated.update(r)
            else:
                aggregated = results
            # Post-process markdown/LaTeX/HTML output
            def postprocess_output(output):
                if isinstance(output, dict):
                    for k, v in output.items():
                        output[k] = postprocess_output(v)
                    return output
                if isinstance(output, list):
                    return [postprocess_output(x) for x in output]
                if isinstance(output, str):
                    # Remove <br/> between block equations
                    output = re.sub(r'(</math>\s*)<br\s*/?>\s*(<math display="block">)', r'\1\2', output)
                    # Replace double brackets with single in LaTeX
                    output = re.sub(r'\[\[([^\]]+)\]\]', r'[\1]', output)
                    # Remove trailing commas in equations
                    output = re.sub(r'(\\\])\s*,', r'\1', output)
                    return output
                return output
            return postprocess_output(aggregated)
        return {}
