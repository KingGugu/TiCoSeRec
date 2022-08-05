# -*- coding: utf-8 -*-

import tqdm
import pandas as pd

"""
Rank users according to the variance of each user's interaction interval. 
Then divide the dataset by half the amount of users.

The output data needs to be run in RecBole (Need to be converted to .inter file).
This code is used for Empirical Study.

Four columns are required in the .csv file: ['user', 'item', 'rate', 'time']
We do not use ratings information, so you can delete this column in the code and CSV file.
"""


def get_var(tlist):
    length = len(tlist)
    total = 0
    diffs = []
    if length == 1:
        return 0

    for i in range(length - 1):
        diff = abs(tlist[i + 1] - tlist[i])
        diffs.append(diff)
        total = total + diff
    avg_diff = total / len(diffs)

    total = 0
    for diff in diffs:
        total = total + (diff - avg_diff) ** 2
    result = total / len(diffs)

    return result


def main(file_in, file_first, file_last):
    df = pd.read_csv(file_in, sep=',', engine='python',
                     names=['user', 'item', 'rate', 'time'])
    # df = pd.read_csv('./Industrial_and_Scientific_unique_idx.csv', sep=',', engine='python')
    # df = df.drop(columns='Unnamed: 0')

    # Multiple interactions at the same time are regarded as one.
    drop_dup = df.drop_duplicates(subset=['user', 'time'], keep='first')
    grouped = drop_dup.groupby(['user'])
    # print(grouped['user'].value_counts())

    vars = []
    users = []
    for group in grouped['time']:
        timelist = []
        users.append(group[0])
        for temp in group[1]:
            timelist.append(temp)
        var = get_var(timelist)
        vars.append(var)

    var_time = pd.DataFrame({'user': users, 'vartime': vars})
    var_time = var_time.sort_values(by=['vartime'])
    users = []
    for user in var_time['user']:
        users.append(user)

    half = int(len(users) / 2)
    first = users[:half]
    last = users[half:]
    df_last = df.copy(deep=True)
    df_first = df.copy(deep=True)

    print('get first 50% users')
    for user in tqdm.tqdm(first):
        df_last.drop(df_last.index[(df_last['user'] == user)], inplace=True)
    print('get last 50% users')
    for user in tqdm.tqdm(last):
        df_first.drop(df_first.index[(df_first['user'] == user)], inplace=True)

    df_first.to_csv(file_first, index=False, header=0)
    df_last.to_csv(file_last, index=False, header=0)


file_name_in = './Beauty.csv'
file_name_first = 'Beauty_first_user.csv'
file_name_last = 'Beauty_last_user.csv'
main(file_name_in, file_name_first, file_name_last)
