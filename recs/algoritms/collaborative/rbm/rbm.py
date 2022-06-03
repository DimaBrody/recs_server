from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from datetime import datetime

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


def dot1(V, W):
    # V is N x D x K (batch of visible units)
    # W is D x K x M (weights)
    # returns N x M (hidden layer size)
    return tf.tensordot(V, W, axes=[[1, 2], [0, 1]])


def dot2(H, W):
    # H is N x M (batch of hiddens)
    # W is D x K x M (weights transposed)
    # returns N x D x K (visible)
    return tf.tensordot(H, W, axes=[[1], [2]])


class RBM(object):
    def __init__(self, D, M, K, saved=None):
        self.D = D  # input feature size
        self.M = M  # hidden size
        self.K = K  # number of ratings
        self.build(D, M, K, saved)

    def build(self, D, M, K, saved):
        # params
        if saved:
            saved.run(tf.compat.v1.global_variables_initializer())
            vars = saved.graph.get_collection("trainable_variables")
            self.W = vars[0]
            self.c = vars[1]
            self.b = vars[2]
        else:
            self.W = tf.Variable(tf.random.normal(shape=(D, K, M)) * np.sqrt(2.0 / M))
            self.c = tf.Variable(np.zeros(M).astype(np.float32))
            self.b = tf.Variable(np.zeros((D, K)).astype(np.float32))

        self.X_in = tf.compat.v1.placeholder(tf.float32, shape=(None, D), name="x_in")

        X = tf.cast(self.X_in - 1, tf.int32)
        X = tf.one_hot(X, K)

        V = X
        p_h_given_v = tf.nn.sigmoid(dot1(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v  # save for later

        r = tf.random.uniform(shape=tf.shape(input=p_h_given_v))
        H = tf.cast(r < p_h_given_v, dtype=tf.float32)


        logits = dot2(H, self.W) + self.b
        cdist = tf.compat.v1.distributions.Categorical(logits=logits)
        X_sample = cdist.sample()
        X_sample = tf.one_hot(X_sample, depth=K)

        mask2d = tf.cast(self.X_in > 0, tf.float32)
        mask3d = tf.stack([mask2d] * K, axis=-1)
        X_sample = X_sample * mask3d

        objective = tf.reduce_mean(input_tensor=self.free_energy(X)) - tf.reduce_mean(
            input_tensor=self.free_energy(X_sample))
        self.train_op = tf.compat.v1.train.AdamOptimizer(1e-2).minimize(objective)

        logits = self.forward_logits(X)
        self.cost = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(X),
                logits=logits,
            )
        )

        self.output_visible = self.forward_output(X)

        # for calculating SSE
        self.one_to_ten = tf.constant((np.arange(10) + 1).astype(np.float32))
        self.pred = tf.tensordot(self.output_visible, self.one_to_ten, axes=[[2], [0]])
        mask = tf.cast(self.X_in > 0, tf.float32)
        se = mask * (self.X_in - self.pred) * (self.X_in - self.pred)
        self.sse = tf.reduce_sum(input_tensor=se)

        # test SSE
        self.X_test = tf.compat.v1.placeholder(tf.float32, shape=(None, D))
        mask = tf.cast(self.X_test > 0, tf.float32)
        tse = mask * (self.X_test - self.pred) * (self.X_test - self.pred)
        self.tsse = tf.reduce_sum(input_tensor=tse)

        initop = tf.compat.v1.global_variables_initializer()
        if not saved:
            self.saver = tf.compat.v1.train.Saver()
            self.session = tf.compat.v1.Session()
        else:
            self.session = saved
            self.saved = saved
        self.session.run(initop)

    def fit(self, X, X_test, epochs=3, batch_sz=256, show_fig=True):
        N, D = X.shape
        n_batches = N // batch_sz

        costs = []
        test_costs = []
        for i in range(epochs):
            t0 = datetime.now()
            print("Цикл:", i)
            X, X_test = shuffle(X, X_test)
            for j in range(n_batches):
                x = X[j * batch_sz:(j * batch_sz + batch_sz)].toarray()

                t0_0 = datetime.now()

                _, c = self.session.run(
                    (self.train_op, self.cost),
                    feed_dict={self.X_in: x}
                )

                # if j % 10 == 0:
                    # print("j / n_batches:", j, "/", n_batches, "cost:", c)
            print("Час розрахунку:", datetime.now() - t0)

            t0 = datetime.now()
            sse = 0
            test_sse = 0
            n = 0
            test_n = 0
            for j in range(n_batches):
                x = X[j * batch_sz:(j * batch_sz + batch_sz)].toarray()
                xt = X_test[j * batch_sz:(j * batch_sz + batch_sz)].toarray()

                n += np.count_nonzero(x)
                test_n += np.count_nonzero(xt)

                sse_j, tsse_j = self.get_sse(x, xt)
                sse += sse_j
                test_sse += tsse_j
            c = sse / n
            ct = test_sse / test_n
            print("СКП тренувань:", c)
            print("СКП тестувань:", ct)
            # print("Час розрахунку:", datetime.now() - t0)
            costs.append(c)
            test_costs.append(ct)
        if show_fig:
            plt.plot(costs, label='СКП тренувань')
            plt.plot(test_costs, label='СКП тестувань')
            plt.xlabel("цикли")
            plt.ylabel("СКП")
            plt.legend()
            plt.show()

        # if not self.saved:
        #     self.saver.save(self.session, './saved2/rbm-model')

    def free_energy(self, V):
        first_term = -tf.reduce_sum(input_tensor=dot1(V, self.b))
        second_term = -tf.reduce_sum(
            # tf.log(1 + tf.exp(tf.matmul(V, self.W) + self.c)),
            input_tensor=tf.nn.softplus(dot1(V, self.W) + self.c),
            axis=1
        )
        return first_term + second_term

    def forward_hidden(self, X):
        return tf.nn.sigmoid(dot1(X, self.W) + self.c)

    def forward_logits(self, X):
        Z = self.forward_hidden(X)
        return dot2(Z, self.W) + self.b

    def forward_output(self, X):
        return tf.nn.softmax(self.forward_logits(X))

    def transform(self, X):
        # accepts and returns a real numpy array
        # unlike forward_hidden and forward_output
        # which deal with tensorflow variables
        return self.session.run(self.p_h_given_v, feed_dict={self.X_in: X})

    def get_visible(self, X):
        return self.session.run(self.output_visible, feed_dict={self.X_in: X})

    def get_sse(self, X, Xt):
        return self.session.run(
            (self.sse, self.tsse),
            feed_dict={
                self.X_in: X,
                self.X_test: Xt,
            })


def train():
    start_path = '../../../data/functional/sparse/'
    A = load_npz(start_path + "ratings_sparse_train.npz")
    A_test = load_npz(start_path + "ratings_sparse_test.npz")

    N, M = A.shape
    print(len(A.getnnz(0)))
    rbm = RBM(M, 50, 10)
    rbm.fit(A, A_test, 20)


def out_forward_hidden(X, W, c, b):
    return tf.nn.sigmoid(dot1(X, W) + c)


def out_forward_logits(X, W, c, b):
    Z = out_forward_hidden(X, W, c, b)
    return dot2(Z, W) + b


def out_forward_output(X, W, c, b):
    return tf.nn.softmax(out_forward_logits(X, W, c, b))


def restore():
    D = 9994
    M = 50
    K = 10
    sess = tf.compat.v1.Session()
    new_saver = tf.compat.v1.train.import_meta_graph('./saved/rbm-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./saved/'))

    vars = sess.graph.get_collection("trainable_variables")

    sess.run(tf.compat.v1.global_variables_initializer())

    trained_vars = [sess.run(v) for v in vars]

    books_dict = {7596: 10, 1077: 9, 3370: 8}

    X = lil_matrix((1, 9994))

    print(X.shape)

    for i in range(D):
        if i in books_dict.keys():
            X[0, i] = books_dict[i]

    print(X)

    # X_in = sess.graph.get_tensor_by_name("x_in:0")
    X_in = tf.compat.v1.placeholder(tf.float32, shape=(None, D))

    # one hot encode X
    # first, make each rating an int

    X_WE = tf.cast(X_in - 1, tf.int32)
    X_WE = tf.one_hot(X_WE, K)

    output = out_forward_output(X_WE, *trained_vars)

    # print(output)
    # visible = sess.run(output, feed_dict={X_in: X})
    # print(visible)

    one_to_ten = tf.constant((np.arange(10) + 1).astype(np.float32))
    pred = tf.tensordot(output, one_to_ten, axes=[[2], [0]])
    mask = tf.cast(X_in > 0, tf.float32)

    p_h_given_v = tf.nn.sigmoid(dot1(X_WE, trained_vars[0]) + trained_vars[1])

    # print(sess.run(p_h_given_v, feed_dict={X_in: 2}))

    print(p_h_given_v)
    # print(sess.run((pred * mask), feed_dict={X_in: X}))


def restore_2():
    D = 9994
    M = 50
    K = 10
    sess = tf.compat.v1.Session()
    new_saver = tf.compat.v1.train.import_meta_graph('./saved/rbm-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./saved/'))

    vars = sess.graph.get_collection("trainable_variables")

    sess.run(tf.compat.v1.global_variables_initializer())

    trained_vars = [sess.run(v) for v in vars]

    rbm = RBM(M, 50, 10, sess)
    print(rbm.get_visible(rbm.X_in))


if __name__ == '__main__':
    train()
