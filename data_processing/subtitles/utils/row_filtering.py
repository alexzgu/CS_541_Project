import pandas as pd


def remove_hemisphere(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decides which set of rows to remove (separated by line>=50 or line<50),
    where the removed set has fewer hiragana characters.
    Args:
        df: DataFrame containing subtitle data.

    Returns: DataFrame with rows removed.
    """

    df = df.copy()

    A = df[(df['line'] >= 0) & (df['line'] < 50)]
    B = df[df['line'] >= 50]
    C = df[df['line'] == -1]

    # deciding which group (A or B) to lump group C with

    if A.shape[0] < B.shape[0]:
        A = pd.concat([A, C])
        c_belongs_with = 'A'
    else:
        B = pd.concat([B, C])
        c_belongs_with = 'B'

    A_count = A['cleaned_token'].apply(len).sum()  # top
    B_count = B['cleaned_token'].apply(len).sum()  # bottom

    # if equal
    if A_count == B_count:
        raise(ValueError("A_count and B_count are equal. I know this is improbable but possible, "
                         "but I decided not to deal with this until becomes an actual issue."
                         "I.e., now it is an issue. Please contact Alex."))

    if A_count < B_count:
        if c_belongs_with == 'A':  # remove A
            df = df[~((df['line'] >= 0) & (df['line'] < 50) & df['overlap'])]
        else:
            df = df[~((df['line'] < 50) & df['overlap'])]
    else: # remove B
        if c_belongs_with == 'B':
            df = df[~(((df['line'] >= 50) | (df['line'] == -1)) & df['overlap'])]
        else:
            df = df[~((df['line'] >= 50) & df['overlap'])]

    return df.reset_index(drop=True)


def compute_overlaps(df) -> pd.Series:
    """
    Compute overlapping subtitle segments in a DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'start' and 'end' columns containing subtitle timings

    Returns:
    pandas.Series: Boolean series indicating whether each row overlaps with any other row
    """
    n = len(df)
    is_overlap = [False] * n

    # For each row i, look ahead to find any overlaps
    for i in range(n):
        current_end = df.iloc[i]['end']
        # Look at subsequent rows until we find one that starts after current ends
        j = i + 1
        while j < n and df.iloc[j]['start'] < current_end:
            # If we found an overlap, mark both rows
            is_overlap[i] = True
            is_overlap[j] = True
            j += 1

    return pd.Series(is_overlap, dtype=bool)
