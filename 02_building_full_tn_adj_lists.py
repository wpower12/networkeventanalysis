"""
Need to follow my list in the notes. Goal is to have (user-id, day) based adjacency information.
"""
import pandas as pd
# from naatools import utils
from tqdm import tqdm
import datetime
import pickle
import os

TEMP_WINDOW = 5
DIR_OUT = "data/prepared/full_temporal_net_01"

DATE_START = "2022-01-02"
DATE_END   = "2022-06-20"

# For "Full" run.
FN_RAWS_CENTRAL = "data/raw/historic_users_just_ids_dates.csv"
FN_RAWS_MENTION = "data/raw/mu_just_uids_and_dates.csv"
df_cu = pd.read_csv(FN_RAWS_CENTRAL, header=0, parse_dates=[1], infer_datetime_format=True, low_memory=False)
df_mu = pd.read_csv(FN_RAWS_MENTION, header=0, parse_dates=[1], infer_datetime_format=True, low_memory=False)

## For testing.
# FN_RAWS_CENTRAL = "data/test/small_cu.csv"  # has ALL the things
# FN_RAWS_MENTION = "data/test/small_mu.csv"   # ONLY has dates/mentioned user ids.
# df_cu = pd.read_csv(FN_RAWS_CENTRAL, header=0, parse_dates=[0], infer_datetime_format=True, low_memory=False)
# df_mu = pd.read_csv(FN_RAWS_MENTION, header=0, parse_dates=[0], infer_datetime_format=True, low_memory=False)


"""
Extracting User-Date nodes and creating a consistent index. 
"""
def aggregate_mn_lists(ll):
    x = set()
    for mn_list in ll:
        for uid in mn_list:
            x.add(uid)
    return list(x)

def str_to_list(s):
    if s == "[]":
        return []
    return s.replace("[", "").replace("]", "").replace('\'', "").split(",")


user_date_pairs = []      # indexed by tn_node_id, pointing to a (user_id, date) pair
udp_set = set()
pair_to_tn_id   = dict()  # to map from a pair to their ID.
user_date_mus   = []      # indexed by tn_node_id, pointing to a list of (user_id, date) pairs?
# for label, df in [("central", df_cu[:2000].copy()), ("mentioned", df_mu[:2000].copy())]:
for label, df in [("central", df_cu), ("mentioned", df_mu)]:
    print(f"processing {label}")
    df['datetime'] = df['datetime'].apply(lambda d: d.date)
    print("parsing mentioned user strings to lists.")
    tqdm.pandas(unit="user")
    df['mentioned_users'] = df['mentioned_users'].apply(str_to_list)
    print("creating user ids.")
    for user_id, user_tweets in tqdm(df.groupby('user_id'), unit="user"):
        df_agg = user_tweets.groupby('datetime').agg(mentioned_users=('mentioned_users', aggregate_mn_lists))
        for row in df_agg.iterrows():
            date = row[0]
            mus  = row[1]
            user_date_pair = f"{user_id},{date}"
            pair_tn_id = len(user_date_pairs)
            user_date_pairs.append(user_date_pair)
            udp_set.add(user_date_pair)
            pair_to_tn_id[user_date_pair] = pair_tn_id
            mu_pairs = [f"{mu},{date}" for mu in mus]
            user_date_mus.append(mu_pairs)

"""
We need to iterate over the user_date_mus list, and try to add each mentioned user-date pair to the list, if its not
already in the list, that is. Then we can go ahead with adding edges, knowing we'll have 'good' indexes for each.
"""
print("generating edges.")
# need to create a couple helper collections to speed up date math.
date_strs = []
start_date = datetime.datetime.strptime(DATE_START, "%Y-%m-%d")
end_date   = datetime.datetime.strptime(DATE_END, "%Y-%m-%d")
for i in range(TEMP_WINDOW, 1, -1):
    date_i = (start_date-datetime.timedelta(days=i)).date()
    date_strs.append(f"{date_i}")

num_days = (end_date-start_date).days+TEMP_WINDOW
for i in range(num_days):
    date_i = (start_date+datetime.timedelta(days=i)).date()
    date_strs.append(f"{date_i}")
date_str_2_id = dict()
for ds_id, ds in enumerate(date_strs):
    date_str_2_id[ds] = ds_id

edges = []
for tn_id, mu_list in enumerate(tqdm(user_date_mus, unit="user-day")):
    # adding central_user->mentioned_user edges on the same day.
    for mu in mu_list:
        if mu not in udp_set:
            mu_node_id = len(user_date_pairs)
            user_date_pairs.append(mu)
            udp_set.add(mu)
            pair_to_tn_id[mu] = mu_node_id
        else:
            mu_node_id = pair_to_tn_id[mu]
        edges.append((tn_id, mu_node_id))

    # adding temporal edges - need to extract dates and use time-deltas to build new pairs.
    user_date_pair = user_date_pairs[tn_id]
    user_id_str, date_str = user_date_pair.split(",")
    cur_date_idx = date_str_2_id[date_str]
    for i in range(TEMP_WINDOW):
        past_date_str = date_strs[cur_date_idx-i]
        past_pair = f"{user_id_str},{past_date_str}"  # TODO - Check that the date is being output the same way,
                                                      #       otherwise we might be missing an entire class of edge.
        if past_pair in udp_set:
            edges.append((tn_id, pair_to_tn_id[past_pair]))

print("saving notes and results")
if not os.path.exists(DIR_OUT):
    os.mkdir(DIR_OUT)

with open(f"{DIR_OUT}/notes.txt", "w") as f:
    f.write(f"using TEMP_WINDOW: {TEMP_WINDOW}\n")
    f.write(f"cu filename: {FN_RAWS_CENTRAL}\n")
    f.write(f"mu filename: {FN_RAWS_MENTION}\n")

pickle.dump(user_date_pairs, open(f"{DIR_OUT}/user_date_pairs.p", "wb"))
pickle.dump(pair_to_tn_id, open(f"{DIR_OUT}/pair_to_tn_id.p", "wb"))
pickle.dump(user_date_mus, open(f"{DIR_OUT}/user_date_mus.p", "wb"))
pickle.dump(edges, open(f"{DIR_OUT}/edges.p", "wb"))
