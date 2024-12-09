import csv
import re
from data_processing.utils.tokens import get_tokens
import pandas as pd

def srt_to_csv(srt_file, csv_file):
    with open(srt_file, 'r') as srt_f, open(csv_file, 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['start', 'end', 'token'])  # header row

        for line in srt_f:
            if re.match(r'\d+$', line.strip()):  # skip line numbers
                continue
            match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', line)
            if match:
                start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
                end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
                start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
                end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
                token = next(srt_f).strip()  # read the token line
                writer.writerow([start_time, end_time, token])

def process_gold_csv(gold_csv_dir, time_limits, tokens_dir, include_extra_tokens=False):
    tokens = get_tokens(tokens_dir)  # set of tokens accepted for use in the model
    # for each csv in the directory, process it
    # the process is the following:
    # 1. read the csv
    # 2. if the corresponding time limit tuple (a,b) has b>0, then remove all rows with end times > b.
    # 3. remove spaces from the token column
    # 4. create an 'exclude' column (boolean) defaulting to False
    # 5. for a row, refer to the below for determining whether to exclude it:
    #    a. if the token is not in the tokens set, exclude the column
    #    b. if the token contains "(x)", exclude the column and remove the "(x)" part
    # 6. if the token contains "<sil>" or "<s>", set the token to "<silence>"
    # 7. write the processed csv to a new file

    for idx, limit in time_limits:
        # open the csv file as df
        df = pd.read_csv(f'{gold_csv_dir}/{idx}.csv')

        # sort start ascending, end ascending
        df = df.sort_values(by=['start', 'end'])

        truncated = False
        if limit > 0:
            truncated = True
            df = df[df['end'] <= limit]
        df['token'] = df['token'].replace(' ', '', regex=True)

        df['exclude'] = False
        silence_set = {'<sil>', '<s>', '<silence>'}
        silence_token = '<silence>'

        if not include_extra_tokens:
            # convert all <bre> and <echo> to silence tokens
            silence_set.add('<bre>')
            silence_set.add('<echo>')
        df['token'] = df['token'].apply(lambda x: silence_token if x in silence_set else x)
        # if the token contains '<s>' in any part of it, replace the entire token with silence_token
        df['token'] = df['token'].apply(lambda x: silence_token if '<s>' in x else x)

        df['token'], df['exclude'] = zip(*df.apply(lambda row: helper_exclude_row(row, tokens), axis=1))


        if include_extra_tokens:
            breathing_token = '$breathing'
            echo_token = '$echo'
            df['token'] = df['token'].apply(lambda x: breathing_token if x == '<bre>' else x)
            df['token'] = df['token'].apply(lambda x: echo_token if x == '<echo>' else x)

        # if truncated:
        #     # add a row at the end with entries (current last end time, -1, '<silence>', False)
        #     last_end = df.iloc[-1]['end']
        #     df = df.append({'start': last_end, 'end': float('inf'), 'token': '<silence>', 'exclude': False}, ignore_index=True)
        #     # reset index
        #     df = df.reset_index(drop=True)

        # round start and end times to 3 decimal places
        df['start'] = df['start'].round(3)
        df['end'] = df['end'].round(3)

        # add silence tokens in gaps between tokens, then merge them

        # Insert silence rows where there are gaps
        df['next_start'] = df['start'].shift(-1)
        df['gap'] = df['next_start'] - df['end']
        df['insert_silence'] = df['gap'] > 0

        silence_rows = df[df['insert_silence']].copy()
        silence_rows['start'] = silence_rows['end']
        silence_rows['end'] = silence_rows['next_start']
        silence_rows['token'] = silence_token
        silence_rows['exclude'] = False

        df = pd.concat([df, silence_rows]).sort_values(by=['start', 'end']).reset_index(drop=True)

        # Collapse adjacent silence rows
        df['is_silence'] = df['token'] == silence_token
        df['silence_group'] = (df['is_silence'] != df['is_silence'].shift(1)).cumsum()

        silence_groups = df[df['is_silence']].groupby('silence_group').agg(
            {'start': 'min', 'end': 'max', 'token': 'first', 'exclude': 'first'})
        silence_groups['token'] = silence_token
        silence_groups['exclude'] = False

        non_silence_rows = df[~df['is_silence']]
        df = pd.concat([non_silence_rows, silence_groups]).sort_values(by=['start', 'end']).reset_index(drop=True)

        # drop the columns we don't need anymore
        df.drop(columns=['next_start', 'gap', 'insert_silence', 'is_silence', 'silence_group'], inplace=True)

        # write the processed df to a new csv
        df.to_csv(f'{gold_csv_dir}/processed/{idx}.csv', index=False)


def helper_exclude_row(row, tokens) -> (str, bool):
    token = row['token']
    if '(x)' in row['token']:
        token = token.replace('(x)', '')
        return token, True
    if token not in tokens:
        return token, True
    return token, False


if __name__ == '__main__':
    indices = {0, 6, 16, 19}
    time_limits = [(0,-1), (6, 99.70), (16, -1), (19, 73)]
    srt_dir = '../../data/processed/manually_edited_srts'
    gold_csv_dir = 'golden_csvs'

    # for idx in indices:
    #     srt_to_csv(f'{srt_dir}/{idx}.srt', f'{gold_csv_dir}/{idx}.csv')

    tokens_dir = '../../data/config/tokens.txt'
    process_gold_csv(gold_csv_dir, time_limits, tokens_dir, include_extra_tokens=True)