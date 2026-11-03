"""Pydantic models describing the input and output data."""

from typing import Any
from pydantic import BaseModel


class ParameterSpec(BaseModel):
    """Type description of a parameter or a return value,"""

    type: str


class FunctionDefinition(BaseModel):
    """A function that the LLM can choose to call."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec | None = None


class PromptItem(BaseModel):
    """A natural-language request to translate into a function call."""

    prompt: str


class FunctionCall(BaseModel):
    """The result for one prompt: which function and with what arguments."""

    prompt: str
    name: str
    parameters: dict[str, Any]
