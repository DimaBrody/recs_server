import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('../shrink/book-ratings-small.csv')

df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

user_to_book_all = df.groupby('userId').bookId.agg(list).to_dict()
book_to_user_all = df.groupby('bookId').userId.agg(list).to_dict()

user_to_book = df_train.groupby('userId').bookId.agg(list).to_dict()
book_to_user = df_train.groupby('bookId').userId.agg(list).to_dict()

user_book_keys = zip(df_train.userId, df_train.bookId)
user_book_to_rating = pd.Series(df_train.rating.values, index=user_book_keys).to_dict()

user_book_keys_test = zip(df_test.userId, df_test.bookId)
user_book_to_rating_test = pd.Series(df_test.rating.values, index=user_book_keys_test).to_dict()

print(user_to_book)

# with open('dict/user_to_book.json', 'wb') as f:
#     pickle.dump(user_to_book, f)
#
# with open('dict/book_to_user.json', 'wb') as f:
#     pickle.dump(book_to_user, f)
#
# with open('dict/user_to_book_all.json', 'wb') as f:
#     pickle.dump(user_to_book_all, f)
#
# with open('dict/book_to_user_all.json', 'wb') as f:
#     pickle.dump(book_to_user_all, f)
#
# with open('dict/user_book_to_rating.json', 'wb') as f:
#     pickle.dump(user_book_to_rating, f)
#
# with open('dict/user_book_to_rating_test.json', 'wb') as f:
#     pickle.dump(user_book_to_rating_test, f)
