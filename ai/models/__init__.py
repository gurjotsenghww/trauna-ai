"""
Tràuna AI — models package

Model backends must implement the BaseModel interface:

    class BaseModel:
        def generate(
            self,
            prompt: str,
            width: int = 1024,
            height: int = 576,
            num_inference_steps: int = 4,
            seed: int | None = None,
        ) -> PIL.Image.Image:
            ...
"""
