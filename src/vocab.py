"""Load the model vocabulary and convert tokens to their real text."""

import json


def bytes_to_unicode() -> dict[int, str]:
    """Build the byte-to-character table used by byte-level BPE.

    Every byte (0-255) gets a visible character. Printable bytes keep
    their own character; the others (space, newline, ...) are moved to
    characters from 256 upwards, so the space becomes 'Ġ'.

    Returns:
        A dictionary mapping each byte value to its visible character.
    """
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table: dict[int, str] = {b: chr(b) for b in visible}
    shift = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + shift)
            shift += 1
    return table


def load_token_texts(path: str) -> list[str]:
    """Load the vocabulary file and return the real text of every token.

    Args:
        path: Path to the vocab.json file of the model.

    Returns:
        A list where position i holds the text of token i.

    Raises:
        ValueError: If the file cannot be read or is not a vocabulary.
    """
    try:
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot load vocabulary {path}: {e}")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"Invalid vocabulary file: {path}")

    char_to_byte = {c: b for b, c in bytes_to_unicode().items()}
    texts = [""] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        raw = bytes(char_to_byte[c] for c in token)
        texts[token_id] = raw.decode("utf-8", errors="replace")
    return texts
