import time
import csv
import pandas as pd
import snscrape.modules.twitter as sntwitter


DF_COLUMNS = ['datetime', 'user_id', 'user_fips', 'tweet_id', 'text',
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


def count_digits(text):
    n_digits = 0
    for c in text:
        if c in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            n_digits += 1
    return n_digits


def get_tweet_features(tweet):
    user = tweet.user
    u_ca = user.created
    t_ca = tweet.date

    u_acct_age = (t_ca - u_ca).days  # bc datetime? is it that easy? yup. wow.
    u_n_followers = user.followersCount if user.followersCount else 0
    u_n_following = user.friendsCount if user.friendsCount else 0
    u_n_favorites = user.favouritesCount if user.favouritesCount else 0
    u_n_lists = user.listedCount if user.listedCount else 0
    u_n_tweets = user.statusesCount if user.statusesCount else 0
    t_n_retweets = tweet.retweetCount
    t_n_hashtags = len(tweet.hashtags) if tweet.hashtags else 0
    t_n_user_mentions = len(tweet.mentionedUsers) if tweet.mentionedUsers else 0
    t_n_urls = len(tweet.outlinks) if tweet.outlinks else 0
    t_n_chars = len(tweet.content)
    t_n_digits = count_digits(tweet.content)
    return [u_acct_age, u_n_followers, u_n_following, u_n_favorites, u_n_lists, u_n_tweets,
            t_n_retweets, t_n_hashtags, t_n_user_mentions, t_n_urls, t_n_chars, t_n_digits]


'''
    Reads in the user list and gathers all the users tweets from that time period. Adds them to a csv file. 
'''
def collect_huts(fn_user_ids, fn_proc_users, fn_out, date_range, max_attempts=1, timeout_len_s=1):
    df_user_ids = pd.read_csv(fn_user_ids)
    df_proc_users = pd.read_csv(fn_proc_users, header=None, dtype={0: int})[0].tolist()

    n_users = len(df_user_ids)
    c_users = 0
    for _, row in df_user_ids.iterrows():
        u_id, _, u_name, u_fips = row

        if u_id in df_proc_users:
            continue

        sns_query = "from:{} since:{} until:{}".format(u_name, date_range[0], date_range[1])
        successful = False
        attempts = 0
        while not successful and attempts < max_attempts:
            try:
                user_tweets = []
                for i, tweet in enumerate(sntwitter.TwitterSearchScraper(sns_query).get_items()):
                    features = get_tweet_features(tweet)
                    row = [tweet.date, u_id, u_fips, tweet.id, tweet.content]
                    row.extend(features)
                    user_tweets.append(row)

                successful = True
                user_df = pd.DataFrame(user_tweets, columns=DF_COLUMNS)
                user_df.to_csv(fn_out, mode='a', index=False, quoting=csv.QUOTE_NONNUMERIC, header=False)

                with open(fn_proc_users, 'a') as f_pu:
                    f_pu.write("{}\n".format(u_id))

            except KeyboardInterrupt:
                raise

            except Exception as e:
                print("caught: {}".format(e))
                print("waiting and retrying")
                time.sleep(timeout_len_s)
                attempts += 1

        c_users += 1
        print("{:5d}/{} users".format(c_users, n_users))
