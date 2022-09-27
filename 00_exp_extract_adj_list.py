import pandas as pd
from tqdm import tqdm
from naatools import utils
import pickle


FN_CU = "data/raw/historic_user_tweets_w_mn.csv"
# FN_MU = "data/raw/mb_raw_mentioned_tweets_POS_EXAMPLES.csv"
FN_MU = "data/raw/mu_just_uids_and_dates.csv"

FN_OUT = "data/prepared/mb_full_adj_list.txt"

df_cu = utils.load_huts_mn_df_from_fn(FN_CU)
df_mu = pd.read_csv(FN_MU,
                    index_col=1,
                    header=0,
                    parse_dates=[1],
                    # names=["datetime", "user_id", "tweet_id", "mentioned_users"],
                    infer_datetime_format=True,
                    low_memory=False)

adj_list = dict()
def extract_mus(row):
    global adj_list
    user_id = row['user_id']
    mus = utils.str_to_list(row['mentioned_users'])

    if user_id not in adj_list:
        adj_list[user_id] = []

    for mu in mus:
        if mu not in adj_list[user_id]:
            adj_list[user_id].append(mu)


tqdm.pandas()
df_cu.progress_apply(extract_mus, axis=1)

tqdm.pandas()
df_mu.progress_apply(extract_mus, axis=1)

# pickle.dump(adj_list, open(FN_OUT, 'wb'))

with open(FN_OUT, 'w') as f:
    for u in adj_list:
        f.write(f"{u}: {adj_list[u]}\n")
