from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import json

from marker.logger import get_logger
from tqdm import tqdm

from marker.processors.llm import BaseLLMSimpleBlockProcessor, BaseLLMProcessor
from marker.schema.document import Document
from marker.services import BaseService

logger = get_logger()


class LLMSimpleBlockMetaProcessor(BaseLLMProcessor):
    """
    A wrapper for simple LLM processors, so they can all run in parallel.
    """

    def __init__(
        self,
        processor_lst: List[BaseLLMSimpleBlockProcessor],
        llm_service: BaseService,
        config=None,
    ):
        super().__init__(llm_service, config)
        self.processors = processor_lst

    def is_mostly_math_latex(self, text: str, threshold: float = 0.7) -> bool:
        """
        Heuristic: If >70% of the characters are math/LaTeX symbols, consider it mostly math/LaTeX.
        """
        if not text or not isinstance(text, str):
            return False
        math_chars = set("$\\^_{}[]()=+-*/|<>%0123456789")
        math_count = sum(bool(c in math_chars)
        return (math_count / max(1, len(text))) > threshold

    def __call__(self, document: Document):
        if not self.use_llm or self.llm_service is None:
            return

        total = sum(
            [len(processor.inference_blocks(document)) for processor in self.processors]
        )
        pbar = tqdm(
            desc="LLM processors running", disable=self.disable_tqdm, total=total
        )

        all_prompts = [
            processor.block_prompts(document) for processor in self.processors
        ]
        pending = []
        futures_map = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for i, prompt_lst in enumerate(all_prompts):
                for prompt in prompt_lst:
                    # Skip LLM post-processing for blocks that are mostly math/LaTeX
                    block = prompt.get("block", None)
                    block_text = getattr(block, "text", "") if block is not None else ""
                    if self.is_mostly_math_latex(block_text):
                        logger.info("Skipping LLM post-processing for mostly math/LaTeX block.")
                        pbar.update(1)
                        continue
                    future = executor.submit(self.get_response, dict(prompt))
                    pending.append(future)
                    futures_map[future] = {"processor_idx": i, "prompt_data": prompt}

            for future in pending:
                try:
                    result = future.result()
                    future_data = futures_map.pop(future)
                    processor: BaseLLMSimpleBlockProcessor = self.processors[
                        future_data["processor_idx"]
                    ]
                    # finalize the result
                    processor(result, future_data["prompt_data"], document)
                except Exception as e:
                    logger.warning(f"Error processing LLM response: {e}")

                pbar.update(1)

        pbar.close()

    def get_response(self, prompt_data: dict):
        # Strict prompt enforcement: add a warning for the LLM to ONLY return valid JSON
        prompt = prompt_data["prompt"]
        if "Return only valid JSON" not in prompt:
            prompt = (
                "Return only valid JSON. Do not include any text, markdown, or LaTeX outside the JSON object. "
                "If you cannot extract valid JSON, return an empty JSON object: {}\n" + prompt
            )
        if self.llm_service is None:
            logger.error("LLM service is not set.")
            return {}
        response = self.llm_service(
            prompt,
            prompt_data["image"],
            prompt_data["block"],
            prompt_data["schema"],
        )
        # Validate JSON output: if not valid, log and return empty dict
        if not response:
            logger.warning("LLM returned no response, skipping block.")
            return {}
        # If response is already a dict, return as is
        if isinstance(response, dict):
            return response
        # Try to parse as JSON
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Groq extracted JSON is not valid: {e}\nRaw: {response}")
            return {}
