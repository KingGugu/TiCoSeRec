# -*- coding: utf-8 -*-

import pandas as pd
import tqdm

"""
Generate txt data files for training TiCoSeRec.

Four columns are required in the .csv file: ['user', 'item', 'rate', 'time']
We do not use ratings information, so you can delete this column in the code and CSV file.

rank_type should be selected between var and org
var: users will be ranked by the variance of the interaction interval.
org: users will maintain the original order.
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


def get_org_rank_users(dataframe):
    users = []
    for user in dataframe['user'].unique():
        users.append(user)

    return users


def get_var_rank_users(dataframe):
    # Multiple interactions at the same time are regarded as one.
    drop_dup = dataframe.drop_duplicates(subset=['user', 'time'], keep='first')
    grouped = drop_dup.groupby(['user'])

    vars = []
    users = []
    for group in grouped['time']:
        timelist = []
        users.append(group[0])
        for temp in group[1]:
            timelist.append(temp)
        var = get_var(timelist)
        vars.append(var)

    var_time = pd.DataFrame({'user': users,
                             'vartime': vars})
    var_time = var_time.sort_values(by=['vartime'])

    users = []
    for user in var_time['user']:
        users.append(user)

    return users


def convert_unique_idx(dataframe, column_name):
    column_dict = {x: i for i, x in enumerate(dataframe[column_name].unique())}
    dataframe[column_name] = dataframe[column_name].apply(column_dict.get)
    dataframe[column_name] = dataframe[column_name].astype('int')
    assert dataframe[column_name].min() == 0
    assert dataframe[column_name].max() == len(column_dict) - 1
    return dataframe, column_dict


def main(dataset, date_file_name, rank_type):
    df = pd.read_csv(date_file_name, sep=',', engine='python', names=['user', 'item', 'rate', 'time'])
    df.sort_values(by='user', axis=0, inplace=True)
    df_copy = df.copy()

    if rank_type == 'org':
        rank_users = get_org_rank_users(df_copy)
    else:
        rank_users = get_var_rank_users(df_copy)

    item_file = open(dataset + '_item_' + rank_type + '_rank.txt', mode='w')
    time_file = open(dataset + '_time_' + rank_type + '_rank.txt', mode='w')
    grouped = df_copy.groupby(['user'])

    for user in tqdm.tqdm(rank_users):
        temp = grouped.get_group(user)
        temp = temp.sort_values(by=['time'])
        temp_copy = temp.copy()

        items = []
        times = []
        for i in range(temp_copy.shape[0]):
            row = temp_copy.iloc[i]
            items.append(int(row['item']))
            times.append(int(row['time']))
        item_line = str(user) + " " + " ".join([str(i) for i in items])
        time_line = str(user) + " " + " ".join([str(i) for i in times])
        item_file.write(item_line + '\n')
        time_file.write(time_line + '\n')

    item_file.close()
    time_file.close()


"""
rank_type should be selected between var and org
var: users will be ranked by the variance of the interaction interval.
org: users will maintain the original order.
"""
rank_type = 'var'
dataset = 'Home'
date_file_name = 'Home.csv'
main(dataset, date_file_name, rank_type)
