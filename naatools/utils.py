import pandas as pd
import torch
import numpy as np
import random
import string
import arff
from nltk.corpus import stopwords

HUTSMN_COLS = [
    "datetime",
    "user_id",
    "tweet_id",
    "text",
    "mentioned_users",
    "sentiment",
    "lemmas",
    "anxiety",
    "u_acct_age",
    "u_n_followers",
    "u_n_following",
    "u_n_favorites",
    "u_n_lists",
    "u_n_tweets",
    "t_n_retweets",
    "t_n_hashtags",
    "t_n_user_mentions",
    "t_n_urls",
    "t_n_chars",
    "t_n_digits"]

ARFF_COL_HEADERS = ["account_age",
                    "no_follower",
                    "no_following",
                    "no_userfavourites",
                    "no_lists",
                    "no_tweets",
                    "no_retweets",
                    "no_hashtag",
                    "no_usermention",
                    "no_urls",
                    "no_char",
                    "no_digits",
                    "tweet_class"]

HUTS_COL_HEADERS = ["u_acct_age",
                    "u_n_followers",
                    "u_n_following",
                    "u_n_favorites",
                    "u_n_lists",
                    "u_n_tweets",
                    "t_n_retweets",
                    "t_n_hashtags",
                    "t_n_user_mentions",
                    "t_n_urls",
                    "t_n_chars",
                    "t_n_digits"]

FULL_HUTS_COL_HEADERS = [
                    "datetime",
                    "user_id",
                    "tweet_id",
                    "text",
                    "u_acct_age",
                    "u_n_followers",
                    "u_n_following",
                    "u_n_favorites",
                    "u_n_lists",
                    "u_n_tweets",
                    "t_n_retweets",
                    "t_n_hashtags",
                    "t_n_user_mentions",
                    "t_n_urls",
                    "t_n_chars",
                    "t_n_digits"]


def str_to_tuple_list(s):
    if s == "[]":
        return []
    tuples = s.replace("[", "").replace("]", "").replace("), (", "):(").split(":")
    tuples = [s.replace("(", "").replace(")", "") for s in tuples]
    tuples = [s.split(", ") for s in tuples]
    tuples = [(int(s[0]), int(s[1])) for s in tuples]
    return tuples


def str_to_list(s):
    if s == "[]":
        return []
    return s.replace("[", "").replace("]", "").replace('\'', "").split(",")


def str_to_dict(s):
    if s == "{}": return {}
    entries = s.replace("{", "").replace("}", "").split(", ")
    ret_dict = {}
    for entry in entries:
        bowid, weight = entry.split(":")
        ret_dict[bowid] = int(weight)
    return ret_dict


def read_text_dict(fn):
    text_dict = {}
    with open(fn, 'r') as f:
        for line in f.readlines():
            raw = line.replace("\n", "", ).split(", ")
            if len(raw) == 2:
                k, v = raw
                text_dict[int(v)] = k  # Because we need id -> string
    return text_dict


def make_clean_lemmas(topic_terms):
    def clean_lemmas(text):
        raw_lemmas = text.replace("]", "").replace("[", "").split(", ")
        good_lemmas = []
        for l in raw_lemmas:
            l = l.replace("'", "")
            if l in topic_terms and len(l) > 0:
                good_lemmas.append(l)
        return good_lemmas

    return clean_lemmas


def anx_dict_from_df(df):
    ret_dict = {}
    for _, row in df.iterrows():
        ret_dict[row['lemma']] = row['anxiety']
    return ret_dict


def read_lexicon_to_dict(fn):
    lex_dict = {}
    with open(fn, 'r') as f:
        f.readline()
        for l in f.readlines():
            _, lemma, _, score = l.split(",")
            score = float(score)
            lex_dict[lemma] = score
    return lex_dict


def stopwords_list():
    stopwords_eng = stopwords.words("english")
    stopwords_eng += string.punctuation
    stopwords_eng += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return stopwords_eng


def load_huts_mn_df_from_fn(fn):
    df_huts = pd.read_csv(fn,
                          names=HUTSMN_COLS,
                          index_col=None,
                          parse_dates=["datetime"],
                          infer_datetime_format=True,
                          low_memory=False)
    return df_huts


def load_tweet_df_from_arff(fn):
    rows = []
    for d in arff.load(fn):
        rows.append(list(d))
    df = pd.DataFrame(rows, columns=ARFF_COL_HEADERS)
    df['spam'] = df['tweet_class'].apply(lambda t: 1 if t == 'spammer' else 0)
    df.drop(columns=['tweet_class'], inplace=True)
    return df


def load_tweet_tensors_from_arff(fn):
    df = load_tweet_df_from_arff(fn)
    return np.hsplit(df.to_numpy(), np.array([12]))


def load_huts_df_from_csv(fn):
    df_huts = pd.read_csv(fn,
                          header=0,
                          parse_dates=[0],
                          infer_datetime_format=True,
                          low_memory=False)
    # df_huts = df_huts[df_huts['user_id'] != 'user_id']
    return df_huts


def load_huts_tensor_from_csv(fn):
    df_huts = load_huts_df_from_csv(fn)
    df_huts = df_huts[HUTS_COL_HEADERS]
    df_huts = df_huts.astype('int')
    return df_huts.to_numpy()


def huts_tensor_from_df(df_huts):
    df_huts = df_huts[HUTS_COL_HEADERS]
    df_huts = df_huts.astype('int')
    return df_huts.to_numpy()


def set_random_seeds(seed_value):
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


def read_afinn(fn):
    afin_lex = dict()
    with open(fn, 'r') as f:
        for line in f.readlines():
            word, valance = line.split("\t")
            valance = int(valance.replace("\n", ""))
            afin_lex[word] = valance
    return afin_lex