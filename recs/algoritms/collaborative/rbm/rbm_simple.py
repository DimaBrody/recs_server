import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
# from ....data import shared_data

rbm = None

#
# def init():
#     global rbm
#
#     size = len(shared_data.book_translations_old_uu)
#     rbm = RBM(size, size, 0.3, 25, 200, is_restore=True)


class RBM(object):

    def __init__(self, input_size, output_size,
                 learning_rate, epochs, batchsize, is_restore=False):
        # Define hyperparameters
        self._input_size = input_size
        self._output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batchsize = batchsize

        # Initialize weights and biases using zero matrices
        if not is_restore:
            self.w = np.zeros([input_size, output_size], dtype=np.float32)
            self.hb = np.zeros([output_size], dtype=np.float32)
            self.vb = np.zeros([input_size], dtype=np.float32)
        else:
            additional_string = "./" if os.getcwd().__contains__("rbm") else "./recs/algoritms/collaborative/rbm/"
            with open(additional_string + 'saved_variables.json', 'rb') as f:
                saved_variables = pickle.load(f)
                self.w = saved_variables["w"]
                self.hb = saved_variables["hb"]
                self.vb = saved_variables["vb"]

    def prob_h_given_v(self, visible, w, hb):
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.compat.v1.random_uniform(tf.shape(probs))))

    def train(self, X):
        _w = tf.compat.v1.placeholder(tf.float32, [self._input_size, self._output_size])
        _hb = tf.compat.v1.placeholder(tf.float32, [self._output_size])
        _vb = tf.compat.v1.placeholder(tf.float32, [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size], dtype=np.float32)
        prv_hb = np.zeros([self._output_size], dtype=np.float32)
        prv_vb = np.zeros([self._input_size], dtype=np.float32)

        cur_w = np.zeros([self._input_size, self._output_size], dtype=np.float32)
        cur_hb = np.zeros([self._output_size], dtype=np.float32)
        cur_vb = np.zeros([self._input_size], dtype=np.float32)

        v0 = tf.compat.v1.placeholder(tf.float32, [None, self._input_size])
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.compat.v1.to_float(tf.shape(v0)[0])
        update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        err = tf.reduce_mean(tf.square(v0 - v1))

        error_list = []

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            for epoch in range(self.epochs):
                for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                    batch = X[start:end]
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print('Цикл: %d' % epoch, 'помилка рекострукції: %f' % error)
                error_list.append(error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb
            return error_list

    def rbm_output(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        _vb = tf.constant(self.vb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        hiddenGen = self.sample_prob(self.prob_h_given_v(input_X, _w, _hb))
        visibleGen = self.sample_prob(self.prob_v_given_h(hiddenGen, _w, _vb))
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            return sess.run(out), sess.run(visibleGen), sess.run(hiddenGen)

    def save_data(self):
        data = {
            "w": self.w,
            "hb": self.hb,
            "vb": self.vb
        }
        additional_string = "./" if os.getcwd().__contains__("rbm") else "./recs/algoritms/collaborative/rbm/"
        with open(additional_string + 'saved_variables.json', 'wb') as f:
            pickle.dump(data, f)


def main():
    print(os.getcwd())
    start_path = '../../../data/functional/sparse/'
    additional_string = start_path if os.getcwd().__contains__("rbm") else "./recs/data/functional/sparse/"

    A = load_npz(additional_string + "ratings_sparse_train.npz").toarray()
    A_test = load_npz(additional_string + "ratings_sparse_test.npz").toarray()

    A = np.float32(A)

    size = len(A[0])

    rbm = RBM(size, size, 0.3, 25, 200, is_restore=False)

    err = rbm.train(A)

    outputX, reconstructedX, hiddenX = rbm.rbm_output(A)

    pd.Series(err).plot(logy=False)
    plt.xlabel("Цикли")
    plt.ylabel("Реконструкційна помилка")
    # print(outputX)

    # rbm.save_data()



# def predict_ratings(user_object):
#     book_translations = shared_data.book_translations_old_uu
#
#     books = {book_translations[k]: v for k, v in user_object.items()}
#     #
#     # start_path = '../../../data/functional/sparse/'
#     # additional_string = start_path if os.getcwd().__contains__("rbm") else "./recs/data/functional/sparse/"
#     #
#     # A = load_npz(additional_string + "ratings_sparse_train.npz").toarray()
#     # A = np.float32(A)
#
#     np_user = np.zeros((1, len(book_translations)))
#
#     for k, v in books.items():
#         np_user[0][k] = v
#
#     np_user = np.float32(np_user)
#
#     outputX, reconstructedX, _ = rbm.rbm_output(np_user)
#
#     predictionsArray = reconstructedX
#     pred_validation = predictionsArray[np_user.nonzero()].flatten()
#     print(predictionsArray)
#     print(pred_validation)
#     book_old_output = {shared_data.book_translations_uu_old[i]: outputX[0][i] for i in range(len(outputX[0]))}
#
#     sorted_books = {k: v for k, v in sorted(book_old_output.items(), key=lambda item: item[1], reverse=True)}
#
#     return [i for i in sorted_books.keys()]


if __name__ == "__main__":
    # print(predict_ratings({136033: 10, 232521: 8, 136639: 8}))
    main()
