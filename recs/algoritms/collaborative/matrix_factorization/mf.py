from __future__ import print_function, division

import os
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from copy import deepcopy

from recs.algoritms.collaborative.matrix_factorization.mf_functions import calculate_user

start_path = '../../../data/functional/dict/'
# print(os.path.join(start_path))

with open(start_path + 'user_to_book_all.json', 'rb') as f:
    user_to_book_all = pickle.load(f)

with open(start_path + 'book_to_user_all.json', 'rb') as f:
    book_to_user_all = pickle.load(f)

with open(start_path + 'user_to_book.json', 'rb') as f:
    user_to_book = pickle.load(f)

with open(start_path + 'book_to_user.json', 'rb') as f:
    book_to_user = pickle.load(f)

with open(start_path + 'user_book_to_rating.json', 'rb') as f:
    user_book_to_rating = pickle.load(f)

with open(start_path + 'user_book_to_rating_test.json', 'rb') as f:
    user_book_to_rating_test = pickle.load(f)

with open('../../../../recs/data/shrink/ratings-book-translation.json', 'rb') as f:
    book_translations_old_uu = pickle.load(f)

user_temp = (300000, {64088: 10, 241433: 7, 19137: 8})
user_id_temp = max(user_to_book_all.keys()) + 1

if user_id_temp not in user_to_book.keys():
    book_to_rating = {book_translations_old_uu[book]: r for book, r in user_temp[1].items()}
    user_to_book[user_id_temp] = [i for i in book_to_rating.keys()]

    for item in user_to_book[user_id_temp]:
        book_to_user[item].append(user_id_temp)
        user_book_to_rating[(user_id_temp, item)] = book_to_rating[item]

N = np.max(list(user_to_book_all.keys())) + 2
m1 = np.max(list(book_to_user_all.keys()))
m2 = np.max([m for (u, m), r in user_book_to_rating.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

print("converting...")
user_to_book_rating = {}
for i, books in user_to_book.items():
    r = np.array([user_book_to_rating[(i, j)] for j in books])
    user_to_book_rating[i] = (books, r)
book_to_user_rating = {}
for j, users in book_to_user.items():
    r = np.array([user_book_to_rating[(i, j)] for i in users])
    book_to_user_rating[j] = (users, r)

book_to_user_rating_test = {}
for (i, j), r in user_book_to_rating_test.items():
    if j not in book_to_user_rating_test:
        book_to_user_rating_test[j] = [[i], [r]]
    else:
        book_to_user_rating_test[j][0].append(i)
        book_to_user_rating_test[j][1].append(r)
for j, (users, r) in book_to_user_rating_test.items():
    book_to_user_rating_test[j][1] = np.array(r)
print("conversion done")

# initialize variables
K = 10  # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(user_book_to_rating.values()))


def get_loss(b2u):
    # d: book_id -> (user_ids, ratings)
    N = 0.
    sse = 0
    for j, (u_ids, r) in b2u.items():
        p = W[u_ids].dot(U[j]) + b[u_ids] + c[j] + mu
        delta = p - r
        sse += delta.dot(delta)
        N += len(r)
    return sse / N


epochs = 20
reg = 5.
train_losses = []
test_losses = []
for epoch in range(epochs):
    print("Цикл:", epoch)
    epoch_start = datetime.now()

    t0 = datetime.now()
    for i in user_to_book.keys():
        calculate_user(i, N, K, W, reg, b, U, c, mu, user_to_book_rating, user_to_book)

    print("W та b - опрацьовані:", datetime.now() - t0)

    # update U and c
    t0 = datetime.now()
    for j in book_to_user.keys():
        try:
            u_ids, r = book_to_user_rating[j]
            matrix = W[u_ids].T.dot(W[u_ids]) + np.eye(K) * reg
            vector = (r - b[u_ids] - c[j] - mu).dot(W[u_ids])
            cj = (r - W[u_ids].dot(U[j]) - b[u_ids] - mu).sum()

            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(book_to_user[j]) + reg)

            # if j % (M // 10) == 0:
            #     print("j:", j, "M:", M)
        except KeyError:
            pass
    print("U та c - опрацьовані:", datetime.now() - t0)
    print("тривалість циклу:", datetime.now() - epoch_start)

    # store train loss
    t0 = datetime.now()
    train_losses.append(get_loss(book_to_user_rating))

    # store test loss
    test_losses.append(get_loss(book_to_user_rating_test))
    print("ціна розрахункудля циклу :", datetime.now() - t0)
    print("втрати тренувань для циклу:", train_losses[-1])
    print("втрати тестувань для циклу:", test_losses[-1])

print("загальні втрати тестувань:", train_losses)
print("загальні втрати тренувань:", test_losses)

with open('mf_data.json', 'wb') as f:
    data = {
        "M": M,
        "W": W,
        "b": b,
        "U": U,
        "c": c,
        "mu": mu,
        "N": N,
        "reg": reg,
        "K": K,
        "ubr": user_to_book_rating
    }

    pickle.dump(data, f)

plt.plot(train_losses, label="загальні втрати тренувань")
plt.plot(test_losses, label="загальні втрати тестувань")
plt.xlabel("цикли")
plt.ylabel("СКП")
plt.legend()
plt.show()
