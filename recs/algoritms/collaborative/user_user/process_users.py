import numpy as np
from sortedcontainers import SortedList

try:
    from .uu_data import averages as avg, neighbors as nb, deviations as dv
    from ....data.shared_data import user_to_book as ub, user_book_to_rating as ubr
except ImportError:
    avg = None
    nb = None
    dv = None
    ub = None
    ubr = None

K = 25
limit = 3


def process_user_outer(user_id, user_to_book=ub, user_book_to_rating=ubr, neighbors=nb, averages=avg, deviations=dv):
    books_i = user_to_book[user_id]
    books_i_set = set(books_i)

    ratings_i = {book: user_book_to_rating[(user_id, book)] for book in books_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {book: (rating - avg_i) for book, rating in ratings_i.items()}

    averages[user_id] = avg_i
    deviations[user_id] = dev_i

    sl = SortedList()
    for j in user_to_book.keys():
        books_j = user_to_book[j]
        books_j_set = set(books_j)
        common_books = (books_i_set & books_j_set)
        if len(common_books) >= limit:
            ratings_j = {book: user_book_to_rating[(j, book)] for book in books_j}
            avg_j = np.mean(list(ratings_j.values()))
            dev_j = {book: (rating - avg_j) for book, rating in ratings_j.items()}

            numerator = sum(float(dev_i[m]) * float(dev_j[m]) for m in common_books)
            sigma_i = np.sqrt(sum(dev_i[m] * dev_i[m] for m in common_books))
            sigma_j = np.sqrt(sum(dev_j[m] * dev_j[m] for m in common_books))

            try:
                w_ij = numerator / (float(sigma_i) * float(sigma_j))
            except ZeroDivisionError:
                w_ij = 0

            sl.add((-w_ij, j))
            if len(sl) > K:
                del sl[-1]

    neighbors[user_id] = sl
