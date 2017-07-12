import os
import html
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

def train_with_reg_cv(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
        C=2**np.arange(-8, 1).astype(np.float), seed=42):
    scores = []
    for i, c in enumerate(C):
        model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i)
        model.fit(trX, trY)
        score = model.score(vaX, vaY)
        scores.append(score)
    c = C[np.argmax(scores)]
    model = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C))
    model.fit(trX, trY)
    nnotzero = np.sum(model.coef_ != 0)
    if teX is not None and teY is not None:
        score = model.score(teX, teY)*100.
    else:
        score = model.score(vaX, vaY)*100.
    return score, c, nnotzero


def load_sst(path):
    data = pd.read_csv(path)
    X = data['sentence'].values.tolist()
    Y = data['label'].values
    return X, Y


def sst_binary(data_dir='data/'):
    """
    Most standard models make use of a preprocessed/tokenized/lowercased version
    of Stanford Sentiment Treebank. Our model extracts features from a version
    of the dataset using the raw text instead which we've included in the data
    folder.
    """
    trX, trY = load_sst(os.path.join(data_dir, 'train_binary_sent.csv'))
    vaX, vaY = load_sst(os.path.join(data_dir, 'dev_binary_sent.csv'))
    teX, teY = load_sst(os.path.join(data_dir, 'test_binary_sent.csv'))
    return trX, vaX, teX, trY, vaY, teY


def find_trainable_variables(key):
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))


def preprocess(text, front_pad='\n ', end_pad=' '):
    text = html.unescape(text)
    text = text.replace('\n', ' ').strip()
    text = front_pad+text+end_pad
    text = text.encode()
    return text


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n // size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


class HParams(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
