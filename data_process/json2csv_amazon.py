import gzip
import tqdm
import pandas as pd

"""
Convert Amazon JSON data file to CSV data file.
"""


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def amazon(total_inter, file_in, file_out):
    names = ['user', 'item', 'rating', 'time']
    df = pd.DataFrame(columns=names)
    pbar = tqdm.tqdm(total=total_inter)
    for line in parse(file_in):
        rev = line['reviewerID']
        asin = line['asin']
        ratings = line['overall']
        time = line['unixReviewTime']
        value = {'user': rev, 'item': asin, 'rating': ratings, 'time': time}
        df = df.append(value, ignore_index=True)
        pbar.update(1)

    df, user_mapping = convert_unique_idx(df, 'user')
    df, item_mapping = convert_unique_idx(df, 'item')
    items = []
    for item in df['item']:
        items.append(item)
    if 0 in items:
        df['item'] = df['item'].apply(lambda x: x + 1)
    users = []
    for user in df['user']:
        users.append(user)
    if 0 in users:
        df['user'] = df['user'].apply(lambda x: x + 1)

    df.to_csv(file_out, index=False, header=0)
    print(df.shape)


total = 296337  # Beauty:198,502  Sports:296,337  Home:551,682
data_name = 'reviews_Sports_and_Outdoors_5.json.gz'  # XXX.json.gz
file_name_out = 'Sports.csv'
amazon(total, data_name, file_name_out)
