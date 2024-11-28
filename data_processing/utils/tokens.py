from typing import Set
import pandas as pd


def exclude_if_not_in_token_set(df: pd.DataFrame, tokens_path: str) -> pd.DataFrame:
    """
    Args:
        df: DataFrame containing token data.
        tokens_path: Path to .txt file containing tokens.

    Returns: DataFrame with rows excluded if their token is not in the token set.
    """

    token_set = get_tokens(tokens_path)

    # for rows that do not have a token in the token set, set its 'exclude' column to True
    df['exclude_token_set'] = ~df['token'].isin(token_set)
    df['exclude'] = df['exclude_token_set'] | df['exclude']
    df = df.drop(columns=['exclude_token_set'])

    return df


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

