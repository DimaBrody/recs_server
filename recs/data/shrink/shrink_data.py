import pickle
import pandas as pd
from collections import Counter
import numpy as np

df = pd.read_csv('../edited/books-ratings-edited.csv')
print("original dataframe size:", len(df))

user_ids_counter = Counter(df.userId)
book_ids_counter = Counter(df.bookId)

n = 10000
b = 5000

user_ids = [u for u, _ in user_ids_counter.most_common(n)]
book_ids = [b for b, _ in book_ids_counter.most_common(b)]

df_small = df[df.userId.isin(user_ids) & df.bookId.isin(book_ids)].copy()

new_user_id_map = {}
i = 0
for old in set(df_small.userId.values):
    new_user_id_map[old] = i
    i += 1

new_book_id_map = {}
j = 0
for old in set(df_small.bookId.values):
    new_book_id_map[old] = j
    j += 1

print("j:", j)

df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis=1)
df_small.loc[:, 'bookId'] = df_small.apply(lambda row: new_book_id_map[row.bookId], axis=1)

print(len(set(df_small.userId.values)))


df_small = df_small.drop(columns=['isbn'])

print(df_small.describe())

print("small dataframe size:", len(df_small))

df_small.to_csv('../shrink/book-ratings-small.csv', index=False)

with open('ratings-user-translation.json', 'wb') as f:
    pickle.dump(new_user_id_map, f)

with open('ratings-book-translation.json', 'wb') as f:
    pickle.dump(new_book_id_map, f)
