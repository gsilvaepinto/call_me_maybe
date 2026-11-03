"""Build the text prompt given to the LLM."""

from src.models import FunctionDefinition

def describe_function(fn: FunctionDefinition) -> str:
    """Describe one function on a single line.

    Args:
        fn: The function to describe.

    Returns:
        A line such as "- fn_add_numbers(a: number, b: number): Add ...".
    """
    params = ", ".join(
        f"{name}: {spec.type}" for name, spec in fn.parameters.items()
    )
    return f"- {fn.name}({params}): {fn.description}"

def build_prompt(functions: list[FunctionDefinition], request: str) -> str:
    """Build the prompt that asks the LLM to choose a function call.

    Args:
        functions: The functions the LLM can choose from.
        request: The natural-language request to translate.

    Returns:
        The full prompt text.
    """
    lines = [
        "You translate a user request into a call to one of the "
        "functions below.",
        "Choose the function that matches the request and take its "
        "arguments from the request.",
        "Copy every value exactly as written in the request, in the "
        "same order, keeping signs such as '-' and decimals.",
        "",
        "Functions:",
    ]
    lines += [describe_function(fn) for fn in functions]
    lines += ["", f"Request: {request}", "Call: "]
    return "\n".join(lines)
