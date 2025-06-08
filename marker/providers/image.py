"""ImageProvider for handling image-based document pages."""

import contextlib
import types
from typing import Self, Optional

# Temporary workaround for missing pdftext.schema.Reference
try:
    from pdftext.schema import Reference as PdftextReference
except ImportError:
    # Define a minimal Reference class as fallback
    from pydantic import BaseModel
    class PdftextReference(BaseModel):
        pass
from PIL import Image
from pydantic import BaseModel

from marker.providers import BaseProvider
from marker.schema.polygon import PolygonBox
from marker.schema.text import Line


class ImageProvider(BaseProvider):
    """Provider for image-based documents (single image per instance).

    Raises:
        ValueError: If the page range is invalid.

    """

    page_range: list[int] | None = None
    image_count: int = 1

    def __init__(self, filepath: str, config: Optional[BaseModel | dict] = None) -> None:
        """Initialize the ImageProvider with a file path and optional config.

        Args:
            filepath: Path to the image file.
            config: Optional configuration dictionary.

        Raises:
            ValueError: If the page range is invalid.

        """
        super().__init__(filepath, config)

        # TODO(@moatasimfarooque): Support multi-image files (e.g., TIFF stacks). See issue #123.  # noqa: FIX002
        self.images = [Image.open(filepath)]
        self.page_lines: dict[int, list[Line]] = {i: [] for i in range(self.image_count)}

        if self.page_range is None:
            self.page_range = list(range(self.image_count))

        if not self.page_range or min(self.page_range) < 0 or max(self.page_range) >= self.image_count:
            msg = (
                f"Invalid page range, values must be between 0 and {self.image_count - 1}.  "
                f"Min of provided page range is {min(self.page_range)} and max is {max(self.page_range)}."
            )
            raise ValueError(msg)

        self.page_bboxes = {
            i: [0.0, 0.0, float(self.images[i].size[0]), float(self.images[i].size[1])] for i in self.page_range
        }

    def __len__(self) -> int:
        """Return the number of images (pages).

        Returns:
            int: The number of images (pages).

        """
        return self.image_count

    def get_images(self, idxs: list[int], dpi: int) -> list[Image.Image]:
        """Return images for the given indices.

        Args:
            idxs: List of image indices.
            dpi: Dots per inch (unused).

        Returns:
            list[Image.Image]: List of PIL Image objects for the given indices.

        """
        return [self.images[i] for i in idxs]

    def get_page_bbox(self, idx: int) -> PolygonBox | None:
        """Return the bounding box for the given page index as a PolygonBox.

        Args:
            idx: Page index.

        Returns:
            PolygonBox | None: PolygonBox for the page, or None if not found.

        """
        bbox = self.page_bboxes.get(idx)
        if bbox:
            return PolygonBox.from_bbox([float(x) for x in bbox])
        return None

    def get_page_lines(self, idx: int) -> list[Line]:
        """Return the lines for the given page index.

        Args:
            idx: Page index.

        Returns:
            list[Line]: List of Line objects for the page.

        """
        return self.page_lines[idx]

    def get_page_refs(self, idx: int) -> list[PdftextReference]:  # noqa: PLR6301, ARG002
        """Return references for the given page index (empty for images).

        Args:
            idx: Page index (unused).

        Returns:
            list[PdftextReference]: Empty list (no references for images).

        Note:
            This method must accept self for compatibility with the base class, even though it is not used.

        """
        return []

    def __enter__(self) -> Self:  # type: ignore[override]
        """Enter the runtime context related to this object.

        Returns:
            Self: self

        """
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None:
        """Exit the runtime context and close images.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.

        """
        for img in self.images:
            with contextlib.suppress(Exception):
                img.close()

    def __del__(self) -> None:
        """Destructor to ensure images are closed."""
        for img in getattr(self, "images", []):
            with contextlib.suppress(Exception):
                img.close()
