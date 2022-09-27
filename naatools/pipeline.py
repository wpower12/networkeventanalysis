from os import listdir, path, mkdir
from os.path import isfile, join
import torch
import random
import pandas as pd
from tqdm import tqdm
import csv
from .processing import create_examples_from_raw_mn_sequences, preprocess_tweets_w_alex, create_user_df_w_aggregated_mn
from .utils import stopwords_list, load_huts_mn_df_from_fn

"""
Takes in a file of raw tweets, preprocesses it with the provided configured NLP tools, and then outputs the
resulting preprocessed data to individual files for each user, containing the sequences of processed data  
over the provided date range. This includes sentiment values, anxiety values, and the mentioned users aggregated
over every tweet within a single day of the provided date range. 

The user files are saved to the provided directory. 

This is intended to be used to process the 'central users' tweets, and then their 'mentioned users' tweets into 
separate directories, so that the next pipeline step can keep track of which to use when building the example data.

NOTE - Currently not doing any spam filtering. Keeping that out until the rest of the pipeline is working. 
"""
def process_raw_tweets_to_user_dfs(fn_raw, tkz, ltz, alx, stz, sent_thresh, date_range, fn_out_raw, dir_save_users,
                                   process_mn=False, limit=None):
    df_raw = load_huts_mn_df_from_fn(fn_raw)
    df_raw = preprocess_tweets_w_alex(df_raw,
                                      tkz,
                                      ltz,
                                      stopwords_list(),
                                      alx,
                                      verbose=True,
                                      sent=stz,
                                      mn=process_mn,  # Only True for the central users. don't need it for Mentioned.
                                      limit=limit)
    df_raw.to_csv(fn_out_raw, index=False, quoting=csv.QUOTE_NONNUMERIC)

    if not path.exists(dir_save_users):
        mkdir(dir_save_users)

    # User Sequences
    users = df_raw['user_id'].unique()
    raw_seq_dfs = {}

    # Raw Sequences for each user
    print("generating", flush=True)
    for u, user in enumerate(tqdm(users, delay=0.25, unit=" user")):
        df_user = df_raw[df_raw['user_id'] == user].copy()
        df_user_seq_raw = create_user_df_w_aggregated_mn(df_user, sent_thresh, date_range)
        df_user_seq_raw.to_csv("{}/sequences_{}.csv".format(dir_save_users, user))
        raw_seq_dfs[user] = df_user_seq_raw


"""
Given a directory of 'central users' and a directory of 'mentioned users' preprocessed tweet data, this method will
construct actual sequence examples of sentiment/anxiety data. 
"""
def generate_exs_from_cu_mu_dirs(user_seq_dir, mu_seq_dir, window_size, anx_threshold, limit=None,
                                 target_col="ave_pos_anx", verbose=True):
    user_files = [f for f in listdir(user_seq_dir) if isfile(join(user_seq_dir, f))]
    mu_files = [f for f in listdir(mu_seq_dir) if isfile(join(mu_seq_dir, f))]

    if limit is not None:
        user_files = user_files[:limit]

    if verbose: print("reading raw user sequences.")

    # Need to build the user_id -> df map.
    user_id_2_df = {}
    mus = []
    for f in tqdm(user_files, delay=0.1, unit="user", disable=(not verbose)):
        user_id = int(f.replace("sequences_", "").replace(".csv", ""))
        user_df = pd.read_csv("{}/{}".format(user_seq_dir, f),
                              parse_dates=[0],
                              infer_datetime_format=True,
                              dtype={"mentioned_users": str})
        user_df['mentioned_users'].fillna("", inplace=True)
        user_id_2_df[user_id] = user_df
        mus.extend(user_df['mentioned_users'].unique())

    if limit is None:
        mu_pbar = tqdm(mu_files, delay=0.1, unit="mentioned user", disable=(not verbose))
    else:
        mus = list(set(mus))
        mu_fns = ["sequences_{}.csv".format(muid) for muid in mus if muid != ""]
        mu_pbar = tqdm(mu_fns, delay=0.1, unit="mentioned user", disable=(not verbose))

    mu_id_2_df = {}
    for f in mu_pbar:
        mu_id = int(f.replace("sequences_", "").replace(".csv", ""))
        mu_df = pd.read_csv("{}/{}".format(mu_seq_dir, f),
                            parse_dates=[0],
                            infer_datetime_format=True,
                            dtype={"mentioned_users": str})
        mu_df['mentioned_users'].fillna("", inplace=True)
        mu_id_2_df[mu_id] = mu_df

    # We will iterate over JUST user_id_2_df for our 'central' users for the examples
    # but the merged dict will be used to fill in the mentioned user network data.
    mu_id_2_df.update(user_id_2_df)
    all_uids_2_df = mu_id_2_df

    if verbose: print("building example graph sequences")
    example_graph_seqs = []
    example_labels = []
    for _, user_id in enumerate(tqdm(user_id_2_df, delay=0.1, unit="user", disable=(not verbose))):
        df_central_user = user_id_2_df[user_id]
        graphs, labels = create_examples_from_raw_mn_sequences(df_central_user,
                                                               user_id,
                                                               all_uids_2_df,
                                                               window_size,
                                                               anx_threshold,
                                                               target_col=target_col)
        example_graph_seqs.extend(graphs)
        example_labels.extend(labels)
    return example_graph_seqs, example_labels


def generate_exs_from_cu_mu_dirs_cached(user_seq_dir, mu_seq_dir, window_size, anx_threshold, limit=None,
                                 target_col="ave_pos_anx", verbose=True):
    user_files = [f for f in listdir(user_seq_dir) if isfile(join(user_seq_dir, f))]
    mu_files = [f for f in listdir(mu_seq_dir) if isfile(join(mu_seq_dir, f))]

    if limit is not None:
        user_files = user_files[:limit]

    if verbose: print("reading raw user sequences.")

    # Need to build the user_id -> df map.
    user_id_2_df = {}
    mus = []
    for f in tqdm(user_files, delay=0.1, unit="user", disable=(not verbose)):
        user_id = int(f.replace("sequences_", "").replace(".csv", ""))
        user_df = pd.read_csv("{}/{}".format(user_seq_dir, f),
                              parse_dates=[0],
                              infer_datetime_format=True,
                              dtype={"mentioned_users": str})
        user_df['mentioned_users'].fillna("", inplace=True)
        user_id_2_df[user_id] = user_df
        mus.extend(user_df['mentioned_users'].unique())

    mu_fns = {mu: "sequences_{}.csv".format(mu) for mu in mus}

    # mu_id_2_df = {}
    # for f in mu_pbar:
    #     mu_id = int(f.replace("sequences_", "").replace(".csv", ""))
    #     mu_df = pd.read_csv("{}/{}".format(mu_seq_dir, f),
    #                         parse_dates=[0],
    #                         infer_datetime_format=True,
    #                         dtype={"mentioned_users": str})
    #     mu_df['mentioned_users'].fillna("", inplace=True)
    #     mu_id_2_df[mu_id] = mu_df

    # We will iterate over JUST user_id_2_df for our 'central' users for the examples
    # but the merged dict will be used to fill in the mentioned user network data.
    # mu_id_2_df.update(user_id_2_df)
    # all_uids_2_df = mu_id_2_df

    if verbose: print("building example graph sequences")
    example_graph_seqs = []
    example_labels = []
    for _, user_id in enumerate(tqdm(user_id_2_df, delay=0.1, unit="user", disable=(not verbose))):
        df_central_user = user_id_2_df[user_id]
        graphs, labels = create_examples_from_raw_mn_sequences(df_central_user,
                                                               user_id,
                                                               mu_fns,
                                                               window_size,
                                                               anx_threshold,
                                                               target_col=target_col)
        example_graph_seqs.extend(graphs)
        example_labels.extend(labels)
    return example_graph_seqs, example_labels


def read_user_mn_examples_from_dir(user_seq_dir, window_size, anx_threshold, limit=None, target_col="ave_pos_anx",
                                   verbose=True):
    user_files = [f for f in listdir(user_seq_dir) if isfile(join(user_seq_dir, f))]

    if limit is not None:
        user_files = user_files[:limit]

    if verbose: print("reading raw user sequences.")

    # Need to build the user_id -> df map.
    user_id_2_df = {}
    for f in tqdm(user_files, delay=0.1, unit="user", disable=(not verbose)):
        user_id = int(f.replace("sequences_", "").replace(".csv", ""))
        user_df = pd.read_csv("{}/{}".format(user_seq_dir, f),
                              parse_dates=[0],
                              infer_datetime_format=True,
                              dtype={"mentioned_users": str})
        user_df['mentioned_users'].fillna("", inplace=True)
        user_id_2_df[user_id] = user_df

    if verbose: print("building example graph sequences")
    example_graph_seqs = []
    example_labels = []
    for _, user_id in enumerate(tqdm(user_id_2_df, delay=0.1, unit="user", disable=(not verbose))):
        df_central_user = user_id_2_df[user_id]
        graphs, labels = create_examples_from_raw_mn_sequences(df_central_user,
                                                               user_id,
                                                               user_id_2_df,
                                                               window_size,
                                                               anx_threshold,
                                                               target_col=target_col)
        example_graph_seqs.extend(graphs)
        example_labels.extend(labels)
    return example_graph_seqs, example_labels


def balance_and_split_data(sequences, labels, test_frac=0.1, limit=None):
    print("balancing dataset and splitting test/train groups.")
    idx_pos = [idx for idx, l in enumerate(labels) if l.sum() > 0]
    idx_neg = [idx for idx, l in enumerate(labels) if l.sum() == 0]
    print("{}/{} raw positive example sequences".format(len(idx_pos), len(labels)))

    # First we split the raw pos and negative examples into properly sized test/train groups
    cutoff_pos = int((1 - test_frac) * len(idx_pos))
    cutoff_neg = int((1 - test_frac) * len(idx_neg))
    train_idx_pos = idx_pos[:cutoff_pos]
    train_idx_neg = idx_neg[:cutoff_neg]
    test_idx_pos = idx_pos[cutoff_pos:]
    test_idx_neg = idx_neg[cutoff_neg:]

    num_copies_pos_train = int(len(train_idx_pos) / len(train_idx_neg))
    num_copies_pos_test = int(len(train_idx_pos) / len(train_idx_neg))

    # Then we actually build the test/train data sets by creating multiple copies of the positive examples.
    train_seqs = []
    train_labels = []
    # Add copies of positive examples
    for i in range(num_copies_pos_train):
        for p_idx in train_idx_pos:
            train_seqs.append(sequences[p_idx])
            train_labels.append(labels[p_idx])
    # Add negative examples
    for n_idx in train_idx_neg:
        train_seqs.append(sequences[n_idx])
        train_labels.append(labels[n_idx])
    train_data = list(zip(train_seqs, train_labels))
    random.shuffle(train_data)

    test_seqs = []
    test_labels = []
    # Add copies of positive examples
    for i in range(num_copies_pos_test):
        for p_idx in test_idx_pos:
            test_seqs.append(sequences[p_idx])
            test_labels.append(labels[p_idx])
    # Add negative examples
    for n_idx in test_idx_neg:
        test_seqs.append(sequences[n_idx])
        test_labels.append(labels[n_idx])
    test_data = list(zip(test_seqs, test_labels))
    random.shuffle(test_data)

    if limit is not None:
        train_data = train_data[:limit]
        test_data = test_data[:limit]

    return train_data, test_data


def eval_batch(model, loss_func, data_batch):
    loss_acc = 0
    for (seq, y) in data_batch:
        loss_acc += loss_func(model(seq), y.float()).item()
    return loss_acc / len(data_batch)


'''
'Manually' batches sequences. I think this is the only way to batch the graph sequences? Idk. Seems to be working. 
'''
def train_batched_graph_model(model, criteria, optimizer, data_train, data_test, log_file=None, batch_size=16,
                              num_epochs=1, verbose=True, test_interval=100):
    for epoch in range(num_epochs):
        exs_pb = tqdm(data_train, delay=0.25, unit="examples", disable=(not verbose))
        running_loss = 0
        b = 0
        batch = []
        for (gs, y) in exs_pb:
            batch.append((gs, y))
            if len(batch) < batch_size:
                continue
            else:
                optimizer.zero_grad()

                # Create batches of examples and targets such that the operations can still be back proped/autograd-ed?
                outputs = [model(ex) for (ex, _) in batch]
                outputs = torch.stack(outputs)

                targets = [target.float() for (_, target) in batch]
                targets = torch.stack(targets)

                loss = criteria(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                b += 1
                if verbose:
                    exs_pb.desc = "epoch: {} ave training loss: {:.4}".format(epoch + 1, running_loss / b)
                batch = []

                if log_file is not None:
                    log_file.write("batch_training_loss, {}, {}\n".format(b, running_loss / b))
                    if b % test_interval == 0:
                        log_file.write("batch_test_loss, {}, {}\n".format(b, eval_batch(model, criteria, data_test)))

        epoch_test_loss = eval_batch(model, criteria, data_test)
        if verbose: print("epoch: {} ave test loss: {:.4}".format(epoch + 1, epoch_test_loss))
        if log_file is not None:
            log_file.write("epoch_test_loss, {}, {}\n".format(epoch + 1, epoch_test_loss))
