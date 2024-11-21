from typing import Set


def get_tokens(tokens_path: str) -> Set[str]:
    """
    Args:
        tokens_path: Path to .txt file containing tokens.

    Returns: Set of tokens.
    """

    with open(tokens_path, 'r') as file:
        data = file.read()

    data = data.replace(' ', '').replace('\t', '')
    data = data.split(',')

    return set(data)
