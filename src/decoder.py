"""Constrained decoding: make the LLM write only valid function calls."""

import numpy as np
from llm_sdk import Small_LLM_Model

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
    # Start with every token forbidden: [-inf, -inf, -inf, -inf, ...]
    scores = np.array(logits, dtype=np.float64)
    
    # Start with every token forbidden: [-inf, -inf, -inf, -inf, ...]
    masked = np.full(len(scores), -np.inf)

    # Put back the real score of the allowed tokens only,
    # e.g. allowed = [1, 3] -> [-inf, 7.5, -inf, 6.0, ...]
    masked[allowed] = scores[allowed]
    return int(np.argmax(masked))


def choose_function_name(
        model: Small_LLM_Model,
        token_texts: list[str],
        input_ids: list[str],
        names: list[str]
) -> str:
    """Let the model write the name of one of the given functions.

    The text written so far is always the beginning of a valid name
    followed by a closing quote. Generation stops when a full name and
    its closing quote have been written.

    Args:
        model: The language model.
        token_texts: The real text of every token.
        input_ids: The tokens of the prompt; the chosen tokens are
            appended to this list.
        names: The names of the available functions.

    Returns:
        The chosen function name.
    """
    # Add a closing quote to every name, so we know when a name is finished,
    # e.g. ['fn_add_numbers"', 'fn_greet"', 'fn_reverse_string"']
    targets = [name + '"' for name in names]

    written = ""
     # Repeat until a full name with its closing quote has been written
    while written not in targets:
        # Keep the ids of the tokens that are not empty and that keep
        # (written + token text) at the beginning of a valid name,
        # e.g. after "fn_": "add", "gre", "greet", "re", "a", ...
        allowed = [
            token_id
            for token_id, text in enumerate(token_texts)
            if text and any(t.startswith(written + text) for t in targets)
        ]
        logits = model.get_logits_from_input_ids(input_ids)
        token = pick_best(logits, allowed)
        input_ids.append(token)
        written += token_texts[token]
    return written[:-1]
