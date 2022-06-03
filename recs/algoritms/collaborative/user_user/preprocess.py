import pickle
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from process_users import process_user_outer

start_path = '../../../data/functional/dict/'
# print(os.path.join(start_path))

# if not os.path.exists(start_path + 'user_to_book.json') or \
#         not os.path.exists(start_path + 'book_to_user.json') or \
#         not os.path.exists(start_path + 'user_book_to_rating.json') or \
#         not os.path.exists(start_path + 'user_book_to_rating_test.json'):
#     import data_new_2.dict.preprocess_book_data

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

N = np.max(list(book_to_user_all.keys())) + 1

m1 = np.max(list(book_to_user_all.keys()))
m2 = np.max([m for (u, m), r in user_book_to_rating_test.items()])
M = max(m1, m2) + 1

K = 25  # number of neighbors we'd like to consider
limit = 2  # number of books users must have in common

neighbors = {}
averages = {}
deviations = {}

keys_len = len(user_to_book.keys())
index = 0

for i in user_to_book.keys():
    index += 1
    process_user_outer(i, user_to_book, user_book_to_rating, neighbors, averages, deviations)

    if index % 1000 == 0:
        progress_difference = 0
        print("Процес: {:.2f}%".format(i / keys_len * 100))

with open('uu_neighbors.json', 'wb') as f:
    pickle.dump(neighbors, f)

with open('uu_deviations.json', 'wb') as f:
    pickle.dump(deviations, f)

with open('uu_averages.json', 'wb') as f:
    pickle.dump(averages, f)
