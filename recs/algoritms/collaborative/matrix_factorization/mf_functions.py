import pickle

import numpy as np

from recs.data import shared_data
import os

error_to_catch = getattr(__builtins__, 'FileNotFoundError', IOError)

try:
    with open(os.getcwd() + "/recs/algoritms/collaborative/matrix_factorization/mf_data.json", "rb") as f:
        data = pickle.load(f)

        _user_to_book_rating = data["ubr"]
        _W = data["W"]
        _b = data["b"]
        _U = data["U"]
        _c = data["c"]
        _mu = data["mu"]
        _N = data["N"]
        _M = data["M"]
        _reg = data["reg"]
        _K = data["K"]
except error_to_catch as e:
    _W = None
    _b = None
    _U = None
    _c = None
    _mu = None
    _N = None
    _M = None
    _reg = None
    _K = None
    _user_to_book_rating = None


def predict(user_id, book_id):
    global _W, _b, _U, _c, _mu
    return _W[user_id].dot(_U[book_id]) + _b[user_id] + _c[book_id] + _mu


def update_for_user(user_id):
    # global _W, _b, _M
    # _W = np.vstack((_W, np.random.randn(_M)))
    # _b = np.vstack((_b, np.zeros(1)))
    calculate_user(user_id)


def calculate_user(user_id, N=_N, K=_K, W=_W, reg=_reg, b=_b, U=_U, c=_c, mu=_mu,
                   user_to_book_rating=_user_to_book_rating, user_to_book=shared_data.user_to_book):
    m_ids, r = user_to_book_rating[user_id]
    matrix = U[m_ids].T.dot(U[m_ids]) + np.eye(K) * reg
    vector = (r - b[user_id] - c[m_ids] - mu).dot(U[m_ids])
    bi = (r - U[m_ids].dot(W[user_id]) - c[m_ids] - mu).sum()

    # set the updates
    W[user_id] = np.linalg.solve(matrix, vector)
    b[user_id] = bi / (len(user_to_book[user_id]) + reg)
    #
    # if user_id % (N // 10) == 0:
    #     print("i:", user_id, "N:", N)
