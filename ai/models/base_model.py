"""
Tràuna AI — models/base_model.py

Abstract base class that all model backends must implement.
Swapping a model requires only implementing this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image


class BaseModel(ABC):
    """
    Abstract model interface.

    Any model used by Tràuna AI must implement this class.
    The rest of the application only depends on BaseModel,
    never on a specific model library.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_inference_steps: int = 4,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt:               Image description.
            width:                Output width in pixels.
            height:               Output height in pixels.
            num_inference_steps:  Denoising steps. More = higher quality, slower.
            seed:                 Optional random seed for reproducibility.

        Returns:
            A PIL Image object (RGB, width x height).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}>'
