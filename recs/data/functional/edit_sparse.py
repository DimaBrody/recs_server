from __future__ import print_function, division
from builtins import range

import numpy as np
import pickle
from sklearn.utils import shuffle
from collections import Counter

import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from datetime import datetime

df = pd.read_csv('../shrink/book-ratings-small.csv')

N = max(df.userId) + 1
M = max(df.bookId) + 1

df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

A = lil_matrix((N, M))
print("Calling :update_train")
count = 0


def update_train(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count) / cutoff))

    i = int(row.userId)
    j = int(row.bookId)
    A[i, j] = row.rating


df_train.apply(update_train, axis=1)

A = A.tocsc()
mask = (A > 0)
save_npz("sparse/ratings_sparse_train.npz", A)

A_test = lil_matrix((N, M))
print("Calling: update_test")
count = 0


def update_test(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count) / len(df_test)))

    i = int(row.userId)
    j = int(row.bookId)
    A_test[i, j] = row.rating


df_test.apply(update_test, axis=1)
A_test = A_test.tocsr()
mask_test = (A_test > 0)
save_npz("sparse/ratings_sparse_test.npz", A_test)
