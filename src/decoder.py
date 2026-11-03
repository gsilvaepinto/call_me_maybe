"""Constrained decoding: make the LLM write only valid function calls."""

from typing import Any

import numpy as np
from llm_sdk import Small_LLM_Model

from src.models import FunctionCall, FunctionDefinition
from src.prompt import build_prompt

NUMBER_CHARS = set("0123456789.-")
MAX_NUMBER_LENGTH = 24
MAX_STRING_TOKENS = 100


def pick_best(logits: list[float], allowed: list[int]) -> int:
    """Return the allowed token with the highest score.

    Every token that is not allowed gets a score of minus infinity,
    so it can never be chosen.

    Args:
        logits: The score of every token, as given by the model.
        allowed: The ids of the tokens that are allowed.

    Returns:
        The id of the chosen token.
    """
    # Convert the scores to a numpy array, e.g. [9.0, 7.5, 3.0, 6.0, ...]
    scores = np.array(logits, dtype=np.float64)

    # Start with every token forbidden: [-inf, -inf, -inf, -inf, ...]
    masked = np.full(len(scores), -np.inf)

    # Put back the real score of the allowed tokens only,
    # e.g. allowed = [1, 3] -> [-inf, 7.5, -inf, 6.0, ...]
    masked[allowed] = scores[allowed]

    # The best score can only belong to an allowed token, e.g. 1
    return int(np.argmax(masked))


def choose_function_name(
    model: Small_LLM_Model,
    token_texts: list[str],
    input_ids: list[int],
    names: list[str],
) -> str:
    """Let the model write one of the given names.

    The text written so far is always the beginning of a valid name
    followed by a closing quote. Generation stops when a full name and
    its closing quote have been written. Works for any list of words,
    e.g. function names or ["true", "false"].

    Args:
        model: The language model.
        token_texts: The real text of every token.
        input_ids: The tokens so far; the chosen tokens are appended.
        names: The names to choose from.

    Returns:
        The chosen name.
    """
    # Add a closing quote to every name, so we know when a name is finished,
    # e.g. ['fn_add_numbers"', 'fn_greet"', 'fn_reverse_string"']
    targets = [name + '"' for name in names]

    # The part of the name written so far (empty at the start)
    written = ""

    # Repeat until a full name with its closing quote has been written
    while written not in targets:

        # Keep the ids of the tokens that are not empty and that keep
        # (written + token text) at the beginning of a valid name,
        # e.g. after "fn_": "add", "gre", "greet", "re", "a", ...
        allowed = []
        for token_id, text in enumerate(token_texts):
            if text == "":
                continue
            attempt = written + text
            for target in targets:
                if target.startswith(attempt):
                    allowed.append(token_id)
                    break

        # Ask the model for the score of every possible next token
        logits = model.get_logits_from_input_ids(input_ids)

        # Choose the best token among the allowed ones, e.g. "greet"
        token = pick_best(logits, allowed)

        # Add the token to the conversation, so the model sees it next time
        input_ids.append(token)

        # Add the token's text to the name, e.g. "fn_" + "greet"
        written += token_texts[token]

    # written is now e.g. 'fn_greet"'; remove the closing quote
    return written[:-1]


def is_number_prefix(text: str) -> bool:
    """Check that a text can be the beginning of a number.

    Args:
        text: The text written so far.

    Returns:
        True if more characters could turn it into a valid number.
    """
    # Only digits, '.' and '-' are allowed
    if not set(text) <= NUMBER_CHARS:
        return False
    # '-' is only allowed as the first character
    if "-" in text[1:]:
        return False
    # At most one '.'
    return text.count(".") <= 1


def is_complete_number(text: str) -> bool:
    """Check that a text is a complete number, e.g. "42" or "-3.5".

    Args:
        text: The text written so far.

    Returns:
        True if the text is a valid number that ends with a digit.
    """
    return is_number_prefix(text) and text[-1:].isdigit()


def generate_number(
    model: Small_LLM_Model,
    token_texts: list[str],
    input_ids: list[int],
    end: str,
) -> float:
    """Let the model write a number, followed by the given end text.

    Args:
        model: The language model.
        token_texts: The real text of every token.
        input_ids: The tokens so far; the chosen tokens are appended.
        end: The text that closes the number: "," or "}".

    Returns:
        The number written by the model, as a float.

    Raises:
        ValueError: If no valid token can be chosen.
    """
    # The part of the number written so far (empty at the start)
    written = ""

    while True:
        allowed = []
        for token_id, text in enumerate(token_texts):
            candidate = written + text
            # The end token is allowed once the number is complete
            if text == end and is_complete_number(written.removeprefix(" ")):
                allowed.append(token_id)
            # Otherwise the token must keep the text a valid number,
            # optionally preceded by one space (e.g. " -7")
            elif (
                text
                and len(candidate) <= MAX_NUMBER_LENGTH
                and is_number_prefix(candidate.removeprefix(" "))
            ):
                allowed.append(token_id)

        if not allowed:
            raise ValueError(f"Cannot continue the number {written!r}")

        # Same as for the name: ask the model, choose the best allowed token
        logits = model.get_logits_from_input_ids(input_ids)
        token = pick_best(logits, allowed)
        input_ids.append(token)

        # If the model chose the end token, the number is finished
        if token_texts[token] == end:
            return float(written)
        written += token_texts[token]


def generate_string(
    model: Small_LLM_Model,
    token_texts: list[str],
    input_ids: list[int],
) -> str:
    """Let the model write the content of a string, up to its closing quote.

    The opening quote must already be in input_ids. Any token that
    contains a quote closes the string; only the part before the quote
    belongs to the value.

    Args:
        model: The language model.
        token_texts: The real text of every token.
        input_ids: The tokens so far; the chosen tokens are appended.

    Returns:
        The text written by the model, without the quotes.
    """
    # Allowed tokens never change for a string, so compute them once.
    # Only the part before the first quote matters: it must not contain
    # a newline or a backslash, e.g. 'hello', '"', '",' and '!"' are ok
    allowed = []
    for token_id, text in enumerate(token_texts):
        before_quote = text.split('"', 1)[0]
        if text and "\n" not in before_quote and "\\" not in before_quote:
            allowed.append(token_id)

    # The tokens of the string written so far
    content_ids: list[int] = []

    while len(content_ids) < MAX_STRING_TOKENS:
        logits = model.get_logits_from_input_ids(input_ids)
        token = pick_best(logits, allowed)
        input_ids.append(token)

        # A token with a quote closes the string: keep what comes before it
        text = token_texts[token]
        if '"' in text:
            return model.decode(content_ids) + text.split('"', 1)[0]
        content_ids.append(token)

    # Too long: stop here
    return model.decode(content_ids)


def add_text(
    model: Small_LLM_Model,
    input_ids: list[int],
    text: str,
) -> None:
    """Append a fixed text to the conversation, without asking the model.

    Args:
        model: The language model (used only to encode the text).
        input_ids: The tokens so far; the new tokens are appended.
        text: The text to add, e.g. ', "parameters": {'.
    """
    input_ids.extend(model.encode(text)[0].tolist())


def generate_value(
    model: Small_LLM_Model,
    token_texts: list[str],
    input_ids: list[int],
    value_type: str,
    end: str,
) -> Any:
    """Let the model write one parameter value of the given type.

    After this function, the end text ("," or "}") is always written.

    Args:
        model: The language model.
        token_texts: The real text of every token.
        input_ids: The tokens so far; the chosen tokens are appended.
        value_type: The type from the function definition, e.g. "number".
        end: The text that follows the value: "," or "}".

    Returns:
        The value, converted to the right Python type.
    """
    # Numbers: generate_number already writes the end text itself
    if value_type in ("number", "float"):
        return generate_number(model, token_texts, input_ids, end)
    if value_type == "integer":
        return int(generate_number(model, token_texts, input_ids, end))

    # Booleans: let the model choose between the two words
    if value_type == "boolean":
        add_text(model, input_ids, '"')
        word = choose_function_name(
            model, token_texts, input_ids, ["true", "false"]
        )
        add_text(model, input_ids, end)
        return word == "true"

    # Strings, and any unknown type, are written as text
    add_text(model, input_ids, ' "')
    text = generate_string(model, token_texts, input_ids)
    add_text(model, input_ids, ' "')
    return text


def generate_call(
    model: Small_LLM_Model,
    token_texts: list[str],
    functions: list[FunctionDefinition],
    request: str,
) -> FunctionCall:
    """Translate one request into a function call.

    Args:
        model: The language model.
        token_texts: The real text of every token.
        functions: The functions the model can choose from.
        request: The natural-language request.

    Returns:
        The chosen function and its arguments.
    """
    # 1. The prompt, followed by the beginning of the answer
    prompt = build_prompt(functions, request) + '{"name": "'
    input_ids = model.encode(prompt)[0].tolist()

    # 2. The model chooses the function name
    names = [fn.name for fn in functions]
    name = choose_function_name(model, token_texts, input_ids, names)
    function = next(fn for fn in functions if fn.name == name)

    # 3. The model writes each parameter, in the order of the definition
    add_text(model, input_ids, ', "parameters": {')
    parameters: dict[str, Any] = {}
    items = list(function.parameters.items())
    for index, (param_name, spec) in enumerate(items):
        # The last parameter is followed by "}", the others by ","
        end = "}" if index == len(items) - 1 else ","
        add_text(model, input_ids, f' "{param_name}":')
        parameters[param_name] = generate_value(
            model, token_texts, input_ids, spec.type, end
        )

    return FunctionCall(prompt=request, name=name, parameters=parameters)
