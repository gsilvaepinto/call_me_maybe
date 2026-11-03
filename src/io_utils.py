"""Reading and writing the JSON files used by the program."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from src.models import FunctionDefinition, FunctionCall, PromptItem


def read_json_list(path: str) -> list[Any]:
    """Read a JSON file and check that it contains a list.

    Args:
        path: Path to the JSON file.

    Returns:
        The list stored in the file.

    Raises:
        ValueError: If the file is missing, unreadable or not a JSON list.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")
    except OSError as e:
        raise ValueError(f"Cannot read {path}: {e}")

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate the function definitions file.

    Args:
        path: Path to the functions definition file.

    Returns:
        The validated function definitions.

    Raises:
        ValueError: If the file or one of its entries is invalid.
    """
    data = read_json_list(path)
    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as e:
        raise ValueError(f"Invalid function definition in {path}:\n{e}")


def load_prompts(path: str) -> list[PromptItem]:
    """Load and validate the prompts file.

    Args:
        path: Path to the prompts file.

    Returns:
        The validated prompts.

    Raises:
        ValueError: If the file or one of its entries is invalid.
    """
    data = read_json_list(path)
    try:
        return [PromptItem.model_validate(item) for item in data]
    except ValidationError as e:
        raise ValueError(f"Invalid prompt in {path}:\n{e}")


def save_results(path: str, results: list[FunctionCall]) -> None:
    """Write the function call results to a JSON file.

    Args:
        path: Path of the output file.
        results: The results to write.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
