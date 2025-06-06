from typing import Optional, List, Annotated

from PIL import Image
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.util import assign_config, verify_config_keys


class BaseService:
    timeout: Annotated[int, "The timeout to use for the service."] = 30
    max_retries: Annotated[
        int, "The maximum number of retries to use for the service."
    ] = 2
    retry_wait_time: Annotated[int, "The wait time between retries."] = 3

    def __init__(self, config: Optional[BaseModel | dict] = None):
        assign_config(self, config)

        # Ensure we have all necessary fields filled out (API keys, etc.)
        verify_config_keys(self)

    def __call__(
        self,
        prompt: str,
        image: Image.Image | List[Image.Image],
        block: Block,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        raise NotImplementedError
