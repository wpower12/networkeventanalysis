import numpy as np
from . import utils

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

DEFAULT_SH_LAYERS = (24, 6)

def load_spam_ham_pipeline(fn_training_data, layer_sizes=None, initial_lr=0.01, max_epochs=200):
    if layer_sizes is None:
        layer_sizes = DEFAULT_SH_LAYERS

    X_gt, y_gt = utils.load_tweet_tensors_from_arff(fn_training_data)  # Ground truth data to train spam model
    sh_clf = make_pipeline(StandardScaler(),
                           MLPClassifier(
                               solver='adam',
                               hidden_layer_sizes=layer_sizes,
                               learning_rate_init=initial_lr,
                               max_iter=max_epochs))
    sh_clf.fit(X_gt, np.ravel(y_gt))
    return sh_clf
