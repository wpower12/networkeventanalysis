import os
import json

from . import utils


def get_raw_dep_trees_from_tweet_csv(fn):
    df = utils.load_huts_df_from_csv(fn)

    temp_in_fn = "tempNAAT.txt"  # Note - dont over complicated and handle labels. just overwrite the same file.
    with open(temp_in_fn, 'w') as f:
        df['text'].apply(lambda t: f.write("{}\n".format(t)))

    return get_raw_dep_trees_from_text(temp_in_fn)


def get_raw_dep_trees_from_text(fn):
    # Shell out to the standford tool.
    depparse_cmd = "java -mx3g edu.stanford.nlp.pipeline.StanfordCoreNLP "
    depparse_cmd += "-annotators tokenize,pos,depparse -file {} -outputFormat json".format(fn)
    os.system(depparse_cmd)

    output_fn = "{}.json".format(fn)
    with open(output_fn, 'r') as f:
        data = json.load(f)

    trees = []
    for sentence in data['sentences']:
        trees.append((sentence['enhancedPlusPlusDependencies'], sentence['tokens']))

    return trees

