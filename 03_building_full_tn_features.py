import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from naatools import pipeline, utils, processing
from tqdm import tqdm

DIR_NET_DATA = "data/prepared/full_temporal_net_04"
FN_USER_DATE_PAIRS = "user_date_pairs.txt"  # List: Indexed by 'tn_node_id' -> User-Date Pair Str
FN_PAIR_2_TN_ID = "pair_to_tn_id.txt"  # Dict: Key(User-Date Pair Str) -> Value(tn_node_id)  # redundant, idk.
FN_PAIR_MU_LISTS = "user_date_mus.txt"  # List: Indexed by 'tn_node_id' -> List of Mentioned User-Dates (pair strs)
FN_EDGES = "edges.txt"  # Unordered List: edge pairs; (SourceNode(tn_node_id), TargetNode(tn_node_id))
FN_OUT_FEATURES = "features.txt"  # List: Indexed by 'tn_node_id' -> list of feature values. Gonna be big.

FN_RAW_CU_TWEETS = "data/raw/historic_user_tweets_w_mn.csv"
FN_RAW_MU_TWEETS = "data/raw/mb_raw_mentioned_tweets.csv"  # Small, to test.
# FN_RAW_MU_TWEETS = "data/raw/mb_raw_mentioned_tweets_POS_EXAMPLES.csv"  # ALL of them. terrible name, sorry future me.

SENT_THRESH = 0.2
AFIN_THRESH = 1
ANX_THRESHOLD = 0.1
DATE_RANGE = ['2022-01-02', '2022-06-20']
DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

# FN_SH_ARFF = "data/raw/spam/95k-continuous.arff"
FN_AFINN_LEX = "data/raw/AFINN-111.txt"
FN_LEXICON = "data/raw/anxiety_lexicon_filtered.csv"

print("loading nlp components")
tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()
stz = SentimentIntensityAnalyzer()
alx = utils.read_lexicon_to_dict(FN_LEXICON)
afn = utils.read_afinn(FN_AFINN_LEX)

pair_2_tnid = dict()
with open(f"{DIR_NET_DATA}/{FN_PAIR_2_TN_ID}", 'r') as f:
    for line in f.readlines():
        pair_str, tnid = line.replace("\n", "").split(":")
        pair_2_tnid[pair_str] = int(tnid)
largest_tnid = max(pair_2_tnid.values())

def aggregate_user_df(df):
    df['datetime'] = df['datetime'].apply(lambda dt: dt.date())
    df['pos_sent'] = df['sentiment'].apply(lambda s: 1 if s > SENT_THRESH else 0)
    df['neg_sent'] = df['sentiment'].apply(lambda s: 1 if s < -1 * SENT_THRESH else 0)

    df['pos_afinn'] = df['sentiment_afinn'].apply(lambda s: 1 if s > AFIN_THRESH else 0)
    df['neg_afinn'] = df['sentiment_afinn'].apply(lambda s: 1 if s < -1 * AFIN_THRESH else 0)

    df['pos_anx'] = df['anxiety'].apply(lambda a: 1 if a > 0.0 else 0)
    df['neg_anx'] = df['anxiety'].apply(lambda a: 1 if a < 0.0 else 0)

    df_agg = df.groupby(['datetime']).agg(
        sum_anxiety=("anxiety", "sum"),
        count_pos_anx=("pos_anx", "sum"),
        count_neg_anx=("neg_anx", "sum"),
        ave_pos_anx=("pos_anx", "mean"),
        ave_neg_anx=("neg_anx", "mean"),
        sum_sentiment=("sentiment", "sum"),
        ave_sentiment=("sentiment", "mean"),
        count_pos_sent=("pos_sent", "sum"),
        count_neg_sent=("neg_sent", "sum"),
        ave_pos_sent=("pos_sent", "mean"),
        ave_neg_sent=("neg_sent", "mean"),
        sum_sentiment_afinn=("sentiment_afinn", "sum"),
        ave_sentiment_afinn=("sentiment_afinn", "mean"),
        count_pos_afinn=("pos_afinn", "sum"),
        count_neg_afinn=("neg_afinn", "sum"),
        ave_pos_afinn=("pos_afinn", "mean"),
        ave_neg_afinn=("neg_afinn", "mean"),
        total_tweets=("sentiment", "count")
    )

    df_agg = df_agg.reset_index()
    df_agg.set_index('datetime', inplace=True)
    return df_agg.reindex(DATE_RANGE, fill_value=0)  # Fills in 'missing' dates


print("creating 'sparse tensor'")
feature_cols = None  # Hacky but w.e
features = [[] for _ in range(largest_tnid+1)]  # TODO - Replace with sparse tensor
for fn_raw in [FN_RAW_CU_TWEETS, FN_RAW_MU_TWEETS]:
    print(f"processing {fn_raw}")
    df_raw = utils.load_huts_mn_df_from_fn(fn_raw)
    df_raw = processing.preprocess_tweets_w_alex(df_raw,
                                                 tkz,
                                                 ltz,
                                                 utils.stopwords_list(),
                                                 alx,
                                                 verbose=True,
                                                 sent=stz,
                                                 afinn=afn)

    users = df_raw['user_id'].unique()
    for u, user in enumerate(tqdm(users, delay=0.25, unit=" user")):
        df_user = df_raw[df_raw['user_id'] == user].copy()
        df_by_date = aggregate_user_df(df_user)
        if feature_cols is None: feature_cols = list(df_by_date.columns)
        for row in df_by_date.iterrows():
            date = row[0].date()
            data = row[1]
            pair_str = f"{user},{date}"
            if pair_str in pair_2_tnid:
                tn_node_id = pair_2_tnid[pair_str]
                features[tn_node_id] = data

print(f"saving features to {DIR_NET_DATA}/{FN_OUT_FEATURES}")
with open(f"{DIR_NET_DATA}/{FN_OUT_FEATURES}", 'w') as f:
    f.write(f"{feature_cols}\n")
    for feature_row in features:
        f.write(f"{list(feature_row)}\n")
