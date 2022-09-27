import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from .utils import str_to_tuple_list, make_clean_lemmas
from tqdm import tqdm
import torch
from torch_geometric.data import Data

from functools import lru_cache

"""
df: A pandas DataFrame with [userid, text] columns
tkz: tokenizer
ltz: lemmatizer
stopwords: list of stopwords.
fn_stub: the string that'll go before [lemma, dict, bow].csv. Assumed to already have its id and stuff.
"""
def preprocess_tweets(df, tkz, ltz, stopwords, out_dir, fn_stub, verbose=False, sent=None):
    clean_urls(df)
    if verbose: print("urls cleaned")

    def calc_sent(text):
        if text is not None:
            return sent.polarity_scores(text)['compound']
        else:
            return 0

    if sent is not None:
        if verbose: print("adding sentiment features")
        df['sentiment'] = df['text'].apply(calc_sent)

    df['text'] = df['text'].apply(make_preprocess(tkz, ltz, stopwords))
    fn_lemma = "{}/{}_{}.csv".format(out_dir, fn_stub, "lemmas")
    if verbose: print("saving lemmas to {}".format(fn_lemma))
    df.to_csv(fn_lemma)

    if verbose: print("building dict.")
    text_dict = Dictionary(df.text)
    fn_dict = "{}/{}_{}.csv".format(out_dir, fn_stub, "dict")
    if verbose: print("saving dict/word indexes to {}".format(fn_dict))
    with open(fn_dict, 'w') as f:
        for k in text_dict.token2id:
            v = text_dict.token2id[k]
            f.write("{}, {}\n".format(k, v))

    if verbose: print("building bow features")
    df['bow_features'] = df['text'].apply(lambda t: text_dict.doc2bow(t))
    fn_bow = "{}/{}_{}.csv".format(out_dir, fn_stub, "bow")
    if verbose: print("saving bow features to {}".format(fn_bow))

    df_save = df.drop(columns=["text", "sentiment"]).copy()
    df_save.to_csv(fn_bow)

    return df, text_dict


""" Same as above, but with the anxiety lexicon (alex)
df: A pandas DataFrame with [userid, text] columns
tkz: tokenizer
ltz: lemmatizer
stopwords: list of stopwords.
alex: Anxiety lexicon: A map of words to their anxiety scores.
fn_stub: the string that'll go before [lemma, dict, bow].csv. Assumed to already have its id and stuff.
"""
def preprocess_tweets_w_alex(df, tkz, ltz, stopwords, alex, verbose=False, sent=None, mn=False, limit=None, afinn=None):
    if limit is not None:
        df = df[:limit].copy()

    clean_urls(df)
    if verbose: print("urls cleaned")

    def calc_sent(text):
        if text is not None:
            return sent.polarity_scores(text)['compound']
        else:
            return 0

    if sent is not None:
        if verbose: print("adding vader sentiment features")
        tqdm.pandas(unit="tweet", delay=0.1, disable=(not verbose))
        df['sentiment'] = df['text'].progress_apply(calc_sent)

    if verbose: print("tokenizing and lemmatizing.")
    tqdm.pandas(unit="tweet", delay=0.1, disable=(not verbose))
    df['lemmas'] = df['text'].progress_apply(make_preprocess(tkz, ltz, stopwords))

    def calc_afinn(lemmas):
        s = 0
        for lemma in lemmas:
            if lemma in afinn:
                s += afinn[lemma]
        return s

    if afinn is not None:
        if verbose: print("adding afinn sentiment features")
        tqdm.pandas(unit="tweet", delay=0.1, disable=(not verbose))
        df['sentiment_afinn'] = df['lemmas'].progress_apply(calc_afinn)

    # calculate anxiety score
    def anxiety_score(lemmas):
        s = 0
        for lemma in lemmas:
            if lemma in alex:
                s += alex[lemma]
        return s

    if verbose: print("adding anxiety score")
    tqdm.pandas(unit="tweet", delay=0.1, disable=(not verbose))
    df['anxiety'] = df['lemmas'].progress_apply(anxiety_score)

    if mn:
        # convert MN string into actual list.
        if verbose: print("processing mentioned user lists")
        tqdm.pandas(unit="tweet", delay=0.1, disable=(not verbose))
        df['mentioned_users'] = df['mentioned_users'].progress_apply(mu_str_to_list)

    return df


def mu_str_to_list(s):
    if s == "[]":
        return []

    raw = s.replace("[", "").replace("]", "")
    return [int(i) for i in raw.split(",")]


def clean_urls(dframe):
    dframe['text'] = dframe['text'].str.replace(r"http\S+", "", regex=True)


def make_preprocess(tkz, ltz, stopwords):
    def preprocess(text):
        if text is not None:
            tokens = tkz.tokenize(text)
            tokens = [t.lower() for t in tokens if t not in stopwords]
            lemmas = [ltz.lemmatize(t) for t in tokens]
            return lemmas
        return []
    return preprocess


def generate_topic_terms(df_bow, text_dict, fn_out, n_topics=50, r_state=1, n_passes=1, verbose=False):
    df_bow.drop(df_bow[df_bow['bow_features'] == "[]"].index, inplace=True) # Drop 'empty' tweets
    df_bow['bow_features'] = df_bow['bow_features'].apply(str_to_tuple_list)

    if verbose: print("bow feature df loaded. performing LDA")
    tweets_lda = LdaModel(df_bow['bow_features'].to_list(),
                          num_topics=n_topics, # This doesn't seem to be working?
                          id2word=text_dict,
                          random_state=r_state,
                          # alpha="auto",
                          passes=n_passes)

    if verbose: print("saving topics to {}".format(fn_out))
    with open("{}".format(fn_out), 'w') as f:
        for topic in tweets_lda.show_topics(num_topics=n_topics, formatted=True):
            f.write("{}\n".format(topic))

    topic_terms = set()
    for topic in tweets_lda.show_topics(num_topics=n_topics, formatted=False):
        terms = topic[1]
        for term in terms:
            topic_terms.add(term[0])

    return topic_terms


def filter_lemmas(df, good_terms):
    df['lemmas'] = df['text'].apply(make_clean_lemmas(good_terms))
    text_dict = Dictionary(df['lemmas'])
    df['bow'] = df['lemmas'].apply(lambda l: text_dict.doc2bow(l))
    return df, text_dict


def add_anxiety_scores(df, anxiety_dict):
    def sum_anxiety(lemmas):
        score = 0.0
        for l in lemmas:
            if l in anxiety_dict:
                score += anxiety_dict[l]
        return score
    df['anxiety'] = df['lemmas'].apply(sum_anxiety)


def filter_terms(df, terms_to_keep):
    def term_filter(text):
        for ft in terms_to_keep:
            if ft in text:
                return True
        return False

    df['filtered'] = df['text'].apply(term_filter)
    return df[df['filtered']]


'''
    Creates a dataframe holding aggregated, rolling sequences of all the values tracked in the sentiment/anxiety 
    project. 
    
    df          - Input dataframe must have columns: datetime (datetime object dtype), sentiment, anxiety
    sent_thresh - Threshold value for labeling a tweet as positive or negative.
    date_range  - (pd daterange object) The extents of the datasets date range, used to fill in 'missing' values/indexes 
    window_size - For the rolling window method, how large the rolling window is. 
'''
def create_rolling_sequences(df, sent_thresh, date_range, window_size):
    df['datetime'] = df['datetime'].apply(lambda dt: dt.date())
    df['pos_sent'] = df['sentiment'].apply(lambda s: 1 if s > sent_thresh else 0)
    df['neg_sent'] = df['sentiment'].apply(lambda s: 1 if s < -1 * sent_thresh else 0)
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
        total_tweets=("sentiment", "count")
    )

    df_agg = df_agg.reset_index()
    df_agg.set_index('datetime', inplace=True)
    df_agg = df_agg.reindex(date_range, fill_value=0)  # Fills in 'missing' dates
    return df_agg.rolling(window=window_size).mean()


def create_user_df_w_aggregated_mn(df, sent_thresh, date_range):
    df['datetime'] = df['datetime'].apply(lambda dt: dt.date())
    df['pos_sent'] = df['sentiment'].apply(lambda s: 1 if s > sent_thresh else 0)
    df['neg_sent'] = df['sentiment'].apply(lambda s: 1 if s < -1 * sent_thresh else 0)
    df['pos_anx'] = df['anxiety'].apply(lambda a: 1 if a > 0.0 else 0)
    df['neg_anx'] = df['anxiety'].apply(lambda a: 1 if a < 0.0 else 0)

    df_agg = df.groupby(['datetime']).agg(
        sum_anxiety=("anxiety", "sum"),
        count_pos_anx=("pos_anx", "sum"),
        count_neg_anx=("neg_anx", "sum"),
        ave_pos_anx=("pos_anx", "mean"),
        ave_neg_anx=("neg_anx", "mean"),
        max_raw_anx=("anxiety", "max"),
        sum_sentiment=("sentiment", "sum"),
        ave_sentiment=("sentiment", "mean"),
        count_pos_sent=("pos_sent", "sum"),
        count_neg_sent=("neg_sent", "sum"),
        ave_pos_sent=("pos_sent", "mean"),
        ave_neg_sent=("neg_sent", "mean"),
        total_tweets=("sentiment", "count"),
        mentioned_users=("mentioned_users", aggregate_mn)
    )

    df_agg = df_agg.reset_index()
    df_agg.set_index('datetime', inplace=True)
    df_agg = df_agg.reindex(date_range, fill_value=0)  # Fills in 'missing' dates
    df_agg["mentioned_users"].replace(0, "\"\"", inplace=True)
    return df_agg


def aggregate_mn(list_of_lists):
    x = ""
    for l in list_of_lists:
        for i in l:
            x += "{},".format(i)
    return x


'''
    Takes a 'full' dataframe of all sequences and creates a dataset of labeled sequences. Each row represents an example
    sequence of length W, where each entry is the aggregated sentiment data from one day. The label is the anxiety value
    for the day immediately following the last day of the sequence. This will provide fodder for us to predict the 
    change in anxiety values from changes in sentiment values. (I think).
    
    df - The dataframe of full rolling sequences. 
    window_size - 
'''
def create_examples_from_full_sequences(df, window_size, anx_threshold, feature_cols=None, target_col='ave_pos_anx'):
    # So maybe this should take a list of which of the columns we want to turn into the features. We have 11 total cols
    # and only want some of them. We could default to X = [ave_pos_sent, ave_neg_sent] and y = [1/0 on anx flags]
    if feature_cols is None:
        feature_cols = ['ave_pos_sent', 'ave_neg_sent']

    df_features = df[feature_cols]
    df_targets  = df[target_col].apply(lambda r: 1 if r > anx_threshold else 0)

    sequences = []
    labels = []
    n_rows = len(df_features)
    for i in range(n_rows):
        if i < (n_rows-window_size-1):
            sequence = []
            for w in range(window_size):
                sequence.append(df_features.iloc[i+w].tolist())

            sequences.append(sequence)
            labels.append(df_targets.iloc[i+window_size].tolist())

    X = np.asarray(sequences)
    y = np.asarray(labels)
    return X, y


def create_examples_from_raw_mn_sequences(df, user, other_users, window_size, anx_threshold, feature_cols=None, target_col='ave_pos_anx'):
    if feature_cols is None:
        feature_cols = ['ave_pos_sent', 'ave_neg_sent']

    df_features = df[feature_cols]
    df_targets  = df[target_col].apply(lambda r: 1 if r > anx_threshold else 0)
    n_rows = len(df_features)

    # We need X, and edge_index for each day. Then we make windows
    graph_sequence = []
    for i in range(n_rows):
        nodeid_2_userid = [user]
        X = list()
        X.append(df_features.iloc[i].tolist())
        edges = [(0, 0)]

        neighbor_str = df.iloc[i]['mentioned_users']
        if neighbor_str != 0:  # If neighbors == 0, there is no list of neighbor_ids. Yay dynamic typing!
            neighbor_ids = [int(s) for s in neighbor_str.split(",")[:-1]]  # If there is a neighbor list, it's a string.
            n_idx = 0
            for _, neighbor in enumerate(neighbor_ids):
                if neighbor in other_users:
                    X.append(other_users[neighbor][feature_cols].iloc[i].tolist())
                    edges.append((0, n_idx))
                    n_idx += 1

        x_t = torch.tensor(X)
        edge_t = torch.tensor(edges)
        edge_t = torch.reshape(edge_t, (2, -1))
        graph_sequence.append(Data(x=x_t, edge_index=edge_t))

    example_sequences = []
    example_labels = []
    for i in range(len(graph_sequence)):
        if i < (n_rows-window_size-1):
            example_sequences.append(graph_sequence[i:i+window_size])

            label = df_targets.iloc[i + window_size].tolist()
            label = torch.tensor([[label]], dtype=torch.long)
            example_labels.append(label)

    return example_sequences, example_labels


def create_examples_from_raw_mn_sequences_cache(df, user, mentioned_user_fns, window_size, anx_threshold, feature_cols=None, target_col='ave_pos_anx'):
    if feature_cols is None:
        feature_cols = ['ave_pos_sent', 'ave_neg_sent']

    df_features = df[feature_cols]
    df_targets  = df[target_col].apply(lambda r: 1 if r > anx_threshold else 0)
    n_rows = len(df_features)

    @lru_cache(maxsize=64)
    def get_other_user(user_id):
        if user_id in mentioned_user_fns:
            mu_fn = mentioned_user_fns[user_id]
            mu_df = pd.read_csv(mu_fn,
                                parse_dates=[0],
                                infer_datetime_format=True,
                                dtype={"mentioned_users": str})
            mu_df['mentioned_users'].fillna("", inplace=True)
            return mu_df
        else:
            return None

    # We need X, and edge_index for each day. Then we make windows
    graph_sequence = []
    for i in range(n_rows):
        # nodeid_2_userid = [user]
        X = list()
        X.append(df_features.iloc[i].tolist())
        edges = [(0, 0)]

        neighbor_str = df.iloc[i]['mentioned_users']
        if neighbor_str != 0:  # If neighbors == 0, there is no list of neighbor_ids. Yay dynamic typing!
            neighbor_ids = [int(s) for s in neighbor_str.split(",")[:-1]]  # If there is a neighbor list, it's a string.
            n_idx = 0
            for _, neighbor in enumerate(neighbor_ids):
                neighbor_features = get_other_user(neighbor)
                if neighbor_features is not None:
                    X.append(neighbor_features)
                    edges.append((0, n_idx))
                    n_idx += 1

        x_t = torch.tensor(X).type(torch.LongTensor)
        edge_t = torch.tensor(edges)
        edge_t = torch.reshape(edge_t, (2, -1))
        graph_sequence.append(Data(x=x_t, edge_index=edge_t))

    example_sequences = []
    example_labels = []
    for i in range(len(graph_sequence)):
        if i < (n_rows-window_size-1):
            example_sequences.append(graph_sequence[i:i+window_size])

            label = df_targets.iloc[i + window_size].tolist()
            label = torch.tensor([[label]])
            example_labels.append(label)

    return example_sequences, example_labels


def extract_cuss_from_gss(gss, verbose=True):
    cuss = []
    if verbose: print("extracting singlue user sequences from graph sequences.")
    for gs in tqdm(gss, unit="sequence", delay=0.25, disable=(not verbose)):
        cus = [graph.x[0] for graph in gs]
        cuss.append(torch.stack(cus, 0))
    return cuss
