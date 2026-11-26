"""Text conditioning for the batch prompt (frozen T5 encoder)."""

from ncntm.models.text.prompt_encoder import PromptEncoder, get_prompt_encoder

__all__ = ["PromptEncoder", "get_prompt_encoder"]
