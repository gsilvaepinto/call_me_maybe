"""Entry point: translate natural-language prompts into function calls."""

import argparse
import sys

from src.io_utils import load_functions, load_prompts, save_results
from src.models import FunctionCall


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
        help="Path to the function definition file.",
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
        print("Error: no functions are definied.", file=sys.stderr)
        return 1

    print(f"Loaded {len(functions)} functions and {len(prompts)} prompts.")
    results: list[FunctionCall] = []

    # TODO (day 3): ask the LLM to translate each prompt

    try:
        save_results(args.output, results)
    except OSError as e:
        print(f"Error: cannot write {args.output}: {e}", file=sys.stderr)
        return 1

    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
