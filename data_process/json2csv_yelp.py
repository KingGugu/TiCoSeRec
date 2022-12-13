import time
import tqdm
import pandas as pd
from numba import jit

"""
Convert Yelp JSON data file to CSV data file.
"""


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def convert_time(date):
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    return time.mktime(timeArray)


def k_core_check(df, u_core, i_core):
    user_counts = df['user'].value_counts()
    to_remove = user_counts[user_counts < u_core].index
    if len(to_remove) > 0:
        return False
    item_counts = df['item'].value_counts()
    to_remove = item_counts[item_counts < i_core].index
    if len(to_remove) > 0:
        return False
    return True


@jit
def get_index(df, to_remove, obj):
    remove_index = []
    for index in tqdm.tqdm(to_remove):
        remove_index.extend(df.index[(df[obj] == index)])
    return remove_index
    

def k_core(df, u_core, i_core):
    is_cored = k_core_check(df, u_core, i_core)
    while not is_cored:
        
        print(df.shape)

        user_counts = df['user'].value_counts()
        to_remove = user_counts[user_counts < u_core].index
        to_remove_index = get_index(df, to_remove, 'user')
        df.drop(index=to_remove_index, inplace=True)
        print(df.shape)

        item_counts = df['item'].value_counts()
        to_remove = item_counts[item_counts < i_core].index
        to_remove_index = get_index(df, to_remove, 'item')
        df.drop(index=to_remove_index, inplace=True)
        print(df.shape)

        is_cored = k_core_check(df, u_core, i_core)

    return df


def parse(path):
    g = open(path, 'r', encoding='utf-8')
    for l in g:
        yield eval(l)


def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


def yelp(data_file, date_min, date_max, rating_score):
    df = getDF(data_file)
    df.drop(labels=['review_id', 'useful', 'funny', 'cool'], axis=1, inplace=True)
    names = ['user', 'item', 'rating', 'time']
    df.drop(df.index[(df['stars'] <= rating_score)], inplace=True)
    df.drop(df.index[(df['date'] < date_min)], inplace=True)
    df.drop(df.index[(df['date'] > date_max)], inplace=True)

    final = pd.DataFrame(columns=names)
    final['user'] = df['user_id']
    final['item'] = df['business_id']
    final['rating'] = df['stars']
    final['time'] = df['date']

    final, user_mapping = convert_unique_idx(final, 'user')
    final, item_mapping = convert_unique_idx(final, 'item')
    items = []
    for item in final['item']:
        items.append(item)
    if 0 in items:
        final['item'] = final['item'].apply(lambda x: x + 1)
    users = []
    for user in final['user']:
        users.append(user)
    if 0 in users:
        final['user'] = final['user'].apply(lambda x: x + 1)

    final['time'] = final['time'].apply(lambda x: convert_time(x))

    final.to_csv('Yelp-B.csv', index=False, header=0)
    return final


date_min = '2019-01-01 00:00:00'
date_max = '2019-12-31 00:00:00'
data_file = 'yelp_academic_dataset_review.json'
file_name_out = './Yelp_5_core.csv'
rating_score = 0
user_core = 5
item_core = 5

datasets = yelp(data_file, date_min, date_max, rating_score)
print(datasets.shape)

# If only run k-core code, no need to run the previous line of code
# datasets = pd.read_csv('Yelp.csv', sep=',', engine='python', names=['user', 'item', 'rate', 'time'])

datasets = k_core(datasets, user_core, item_core)
datasets.to_csv(file_name_out, index=False, header=0)
print(datasets.shape)
