def is_hira(char: str) -> bool:
    """
    Determines if a character is hiragana.
    Args:
        char: Character to check.
    Returns: Whether the character is hiragana.
    """
    return '\u3041' <= char <= '\u3094'


def is_kata(char: str) -> bool:
    """
    Determines if a character is katakana.
    Args:
        char: Character to check.
    Returns: Whether the character is katakana.
    """
    return '\u30A1' <= char <= '\u30F4'


def is_long_vowel(char: str) -> bool:
    """
    Determines if a character is a long vowel sound.
    Args:
        char: Character to check.
    Returns: Whether the character is a long vowel sound.
    """
    return char == '\u30FC'


def to_kana(token: str, include_katakana: bool = False) -> str:
    """
    Filters out non-Japanese characters from a token.
    Args:
        token: Token to filter.
        include_katakana:  Whether to also include katakana characters.
    Returns: Filtered token; by default, this includes just hiragana and the long vowel sound.
    """

    if include_katakana:
        return ''.join([char for char in token if is_hira(char) or is_kata(char) or is_long_vowel(char)])
    return ''.join([char for char in token if is_hira(char) or is_long_vowel(char)])


def to_alphanumeric(token: str) -> str:
    """
    Filters for alphanumeric characters.
    Args:
        token: Token to filter.
    Returns: Filtered token; this contains only alphanumeric symbols.
    """
    return ''.join([char for char in token if char.isalnum()])


def kana_to_hira(token: str) -> str:
    """
    Converts katakana to hiragana.
    Args:
        token: Token to convert.
    Returns: Converted token.
    """
    return ''.join([chr(ord(char) - 96) if is_kata(char) else char for char in token])
