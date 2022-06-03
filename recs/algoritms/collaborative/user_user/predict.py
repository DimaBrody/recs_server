from .uu_data import averages, deviations, neighbors
from ....data import shared_data
from random import randrange
from .process_users import process_user_outer
from .uu_data import averages
from ....models import Book, Rating
from django.forms.models import model_to_dict


# MY_ID = 300000
# [81958, 234002, 47570]

def for_user(user_id, num):
    # (userId -> {bookId: rating})
    # print(book_translations_old_uu[18174])

    ratings = [model_to_dict(rating) for rating in Rating.objects.filter(user_id=user_id)]
    book_to_rating = {rating["book_id"]: rating["rating"] for rating in ratings}

    user_to_book = shared_data.user_to_book
    book_to_user = shared_data.book_to_user
    book_translations_old_uu = shared_data.book_translations_old_uu
    book_translations_uu_old = shared_data.book_translations_uu_old
    user_book_to_rating = shared_data.user_book_to_rating

    if user_id not in user_to_book.keys():
        book_to_rating = {book_translations_old_uu[int(i)]: r for i, r in book_to_rating.items()}
        user_to_book[user_id] = [i for i in book_to_rating.keys()]

        for item in user_to_book[user_id]:
            book_to_user[item].append(user_id)
            user_book_to_rating[(user_id, item)] = book_to_rating[item]

    if user_id not in averages.keys():
        process_user_outer(user_id)

    predicted_books = {}
    for item in book_to_user.keys():
        if item not in user_to_book[user_id]:
            predicted_books[item] = predict(user_id, item)

    predicted_books_sorted = [(book_translations_uu_old[k], v) for k, v in
                              sorted(predicted_books.items(), key=lambda item: item[1], reverse=True)]

    # predicted_books_sorted_kv = {predicted_books_sorted[i]: i for i in range(len(predicted_books_sorted))}
    #
    # books_from_db = [model_to_dict(item) for item in Book.objects.filter(id__in=predicted_books_sorted)]
    #
    # for book in books_from_db:
    #     book["rank"] = predicted_books_sorted_kv[book["id"]]
    #
    # books_from_db = [book for book in sorted(books_from_db, key=lambda d: d["rank"])]
    # for book in books_from_db:
    #     del book["rank"]

    return predicted_books_sorted[:num]


def predict(i, m, nb=neighbors, dv=deviations, avg=averages):
    numerator = 0
    denominator = 0
    for neg_w, j in nb[i]:
        try:
            numerator += -neg_w * dv[j][m]
            denominator += abs(neg_w)
        except KeyError:
            pass
    if denominator == 0:
        prediction = avg[i]
    else:
        prediction = numerator / denominator + float(avg[i])
    prediction = min(10, prediction)
    prediction = max(1, prediction)
    return prediction
