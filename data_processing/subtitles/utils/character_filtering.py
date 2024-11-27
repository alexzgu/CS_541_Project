import pandas as pd

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


def filter_long_vowels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean tokens by removing 'ー' characters and handling empty tokens.

    Parameters:
    df (pd.DataFrame): Input DataFrame with 'token', 'start', and 'end' columns

    Returns:
    pd.DataFrame: Processed DataFrame with tokens cleaned and potentially rows removed
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()

    # Remove 'ー' from tokens
    cleaned_df['token'] = cleaned_df['token'].str.replace('ー', '')

    # Identify rows to remove (empty tokens after cleaning)
    rows_to_remove = cleaned_df['token'].str.len() == 0

    # If any rows need to be removed
    if rows_to_remove.any():
        # Identify indices of rows to remove
        remove_indices = cleaned_df[rows_to_remove].index

        # For each row to be removed, update the previous row's end
        for idx in remove_indices:
            if idx > 0:
                prev_idx = cleaned_df.index[cleaned_df.index < idx][-1]
                cleaned_df.at[prev_idx, 'end'] = cleaned_df.at[idx, 'end']

        # Remove the empty token rows
        cleaned_df = cleaned_df[~rows_to_remove]

    return cleaned_df.reset_index(drop=True)
