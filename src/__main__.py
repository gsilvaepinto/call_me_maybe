"""Entry point: translate natural-language prompts into function calls."""

import argparse
import sys

from llm_sdk import Small_LLM_Model

from src.decoder import generate_call
from src.io_utils import load_functions, load_prompts, save_results
from src.models import FunctionCall
from src.vocab import load_token_texts


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural-language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the prompts file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path of the output file.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the program.

    Returns:
        The exit code: 0 on success, 1 on error.
    """
    args = parse_args()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not functions:
        print("Error: no functions are defined.", file=sys.stderr)
        return 1

    print(f"Loaded {len(functions)} functions and {len(prompts)} prompts.")

    # Load the model and the text of every token
    try:
        model = Small_LLM_Model()
        token_texts = load_token_texts(model.get_path_to_vocab_file())
    except Exception as e:
        print(f"Error: cannot load the model: {e}", file=sys.stderr)
        return 1

    # Translate each prompt; a failing prompt is reported and skipped
    results: list[FunctionCall] = []
    for item in prompts:
        try:
            call = generate_call(model, token_texts, functions, item.prompt)
        except Exception as e:
            print(f"Error on prompt {item.prompt!r}: {e}", file=sys.stderr)
            continue
        results.append(call)
        print(f"{item.prompt!r} -> {call.name}({call.parameters})")

    try:
        save_results(args.output, results)
    except OSError as e:
        print(f"Error: cannot write {args.output}: {e}", file=sys.stderr)
        return 1

    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())