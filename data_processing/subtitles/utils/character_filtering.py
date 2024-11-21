def to_kana(token: str, include_katakana: bool = False) -> str:
    """
    Filters out non-Japanese characters from a token.
    Args:
        token: Token to filter.
        include_katakana:  Whether to include katakana characters.
    Returns: Filtered token; by default, this includes just hiragana.
    """
    if include_katakana:
        return ''.join([char for char in token if '\u3040' <= char <= '\u30FF'])
    return ''.join([char for char in token if '\u3040' <= char <= '\u309F'])

# TODO: do something about ・and ー.
