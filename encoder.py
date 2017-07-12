import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.externals import joblib

from utils import HParams, preprocess, iter_data

global nloaded
nloaded = 0


def load_params(shape, dtype, *args, **kwargs):
    global nloaded
    nloaded += 1
    return params[nloaded - 1]


def embd(X, ndim, scope='embedding'):
    with tf.variable_scope(scope):
        embd = tf.get_variable(
            "w", [hps.nvocab, ndim], initializer=load_params)
        h = tf.nn.embedding_lookup(embd, X)
        return h


def fc(x, nout, act, wn=False, bias=True, scope='fc'):
    with tf.variable_scope(scope):
        nin = x.get_shape()[-1].value
        w = tf.get_variable("w", [nin, nout], initializer=load_params)
        if wn:
            g = tf.get_variable("g", [nout], initializer=load_params)
        if wn:
            w = tf.nn.l2_normalize(w, dim=0) * g
        z = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("b", [nout], initializer=load_params)
            z = z+b
        h = act(z)
        return h


def mlstm(inputs, c, h, M, ndim, scope='lstm', wn=False):
    nin = inputs[0].get_shape()[1].value
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, ndim * 4], initializer=load_params)
        wh = tf.get_variable("wh", [ndim, ndim * 4], initializer=load_params)
        wmx = tf.get_variable("wmx", [nin, ndim], initializer=load_params)
        wmh = tf.get_variable("wmh", [ndim, ndim], initializer=load_params)
        b = tf.get_variable("b", [ndim * 4], initializer=load_params)
        if wn:
            gx = tf.get_variable("gx", [ndim * 4], initializer=load_params)
            gh = tf.get_variable("gh", [ndim * 4], initializer=load_params)
            gmx = tf.get_variable("gmx", [ndim], initializer=load_params)
            gmh = tf.get_variable("gmh", [ndim], initializer=load_params)

    if wn:
        wx = tf.nn.l2_normalize(wx, dim=0) * gx
        wh = tf.nn.l2_normalize(wh, dim=0) * gh
        wmx = tf.nn.l2_normalize(wmx, dim=0) * gmx
        wmh = tf.nn.l2_normalize(wmh, dim=0) * gmh

    cs = []
    for idx, x in enumerate(inputs):
        m = tf.matmul(x, wmx)*tf.matmul(h, wmh)
        z = tf.matmul(x, wx) + tf.matmul(m, wh) + b
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        if M is not None:
            ct = f*c + i*u
            ht = o*tf.tanh(ct)
            m = M[:, idx, :]
            c = ct*m + c*(1-m)
            h = ht*m + h*(1-m)
        else:
            c = f*c + i*u
            h = o*tf.tanh(c)
        inputs[idx] = h
        cs.append(c)
    cs = tf.stack(cs)
    return inputs, cs, c, h


def model(X, S, M=None, reuse=False):
    nsteps = X.get_shape()[1]
    cstart, hstart = tf.unstack(S, num=hps.nstates)
    with tf.variable_scope('model', reuse=reuse):
        words = embd(X, hps.nembd)
        inputs = tf.unstack(words, nsteps, 1)
        hs, cells, cfinal, hfinal = mlstm(
            inputs, cstart, hstart, M, hps.nhidden, scope='rnn', wn=hps.rnn_wn)
        hs = tf.reshape(tf.concat(hs, 1), [-1, hps.nhidden])
        logits = fc(
            hs, hps.nvocab, act=lambda x: x, wn=hps.out_wn, scope='out')
    states = tf.stack([cfinal, hfinal], 0)
    return cells, states, logits


def ceil_round_step(n, step):
    return int(np.ceil(n/step)*step)


def batch_pad(xs, nbatch, nsteps):
    xmb = np.zeros((nbatch, nsteps), dtype=np.int32)
    mmb = np.ones((nbatch, nsteps, 1), dtype=np.float32)
    for i, x in enumerate(xs):
        l = len(x)
        npad = nsteps-l
        xmb[i, -l:] = list(x)
        mmb[i, :npad] = 0
    return xmb, mmb


class Model(object):

    def __init__(self, nbatch=128, nsteps=64):
        global hps
        hps = HParams(
            load_path='model_params/params.jl',
            nhidden=4096,
            nembd=64,
            nsteps=nsteps,
            nbatch=nbatch,
            nstates=2,
            nvocab=256,
            out_wn=False,
            rnn_wn=True,
            rnn_type='mlstm',
            embd_wn=True,
        )
        global params
        params = [np.load('model/%d.npy'%i) for i in range(15)]
        params[2] = np.concatenate(params[2:6], axis=1)
        params[3:6] = []

        X = tf.placeholder(tf.int32, [None, hps.nsteps])
        M = tf.placeholder(tf.float32, [None, hps.nsteps, 1])
        S = tf.placeholder(tf.float32, [hps.nstates, None, hps.nhidden])
        cells, states, logits = model(X, S, M, reuse=False)

        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)

        def seq_rep(xmb, mmb, smb):
            return sess.run(states, {X: xmb, M: mmb, S: smb})

        def seq_cells(xmb, mmb, smb):
            return sess.run(cells, {X: xmb, M: mmb, S: smb})

        def transform(xs):
            tstart = time.time()
            xs = [preprocess(x) for x in xs]
            lens = np.asarray([len(x) for x in xs])
            sorted_idxs = np.argsort(lens)
            unsort_idxs = np.argsort(sorted_idxs)
            sorted_xs = [xs[i] for i in sorted_idxs]
            maxlen = np.max(lens)
            offset = 0
            n = len(xs)
            smb = np.zeros((2, n, hps.nhidden), dtype=np.float32)
            for step in range(0, ceil_round_step(maxlen, nsteps), nsteps):
                start = step
                end = step+nsteps
                xsubseq = [x[start:end] for x in sorted_xs]
                ndone = sum([x == b'' for x in xsubseq])
                offset += ndone
                xsubseq = xsubseq[ndone:]
                sorted_xs = sorted_xs[ndone:]
                nsubseq = len(xsubseq)
                xmb, mmb = batch_pad(xsubseq, nsubseq, nsteps)
                for batch in range(0, nsubseq, nbatch):
                    start = batch
                    end = batch+nbatch
                    batch_smb = seq_rep(
                        xmb[start:end], mmb[start:end],
                        smb[:, offset+start:offset+end, :])
                    smb[:, offset+start:offset+end, :] = batch_smb
            features = smb[0, unsort_idxs, :]
            print('%0.3f seconds to transform %d examples' %
                  (time.time() - tstart, n))
            return features

        def cell_transform(xs, indexes=None):
            Fs = []
            xs = [preprocess(x) for x in xs]
            for xmb in tqdm(
                    iter_data(xs, size=hps.nbatch), ncols=80, leave=False,
                    total=len(xs)//hps.nbatch):
                smb = np.zeros((2, hps.nbatch, hps.nhidden))
                n = len(xmb)
                xmb, mmb = batch_pad(xmb, hps.nbatch, hps.nsteps)
                smb = sess.run(cells, {X: xmb, S: smb, M: mmb})
                smb = smb[:, :n, :]
                if indexes is not None:
                    smb = smb[:, :, indexes]
                Fs.append(smb)
            Fs = np.concatenate(Fs, axis=1).transpose(1, 0, 2)
            return Fs

        self.transform = transform
        self.cell_transform = cell_transform


if __name__ == '__main__':
    mdl = Model()
    text = ['demo!']
    text_features = mdl.transform(text)
    print(text_features.shape)
