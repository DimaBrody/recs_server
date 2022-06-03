import pickle

book_translations_old_uu = {}
book_translations_uu_old = {}
user_to_book = {}
book_to_user = {}
user_book_to_rating = {}


def setup_data():
    global book_translations_old_uu
    global book_translations_uu_old

    global user_to_book
    global book_to_user
    global user_book_to_rating

    book_translations_uu_old.clear()
    book_translations_old_uu.clear()
    user_to_book.clear()
    book_to_user.clear()
    user_book_to_rating.clear()

    with open('./recs/data/shrink/ratings-book-translation.json', 'rb') as f:
        book_translations_old_uu = pickle.load(f)

    start_path = "./recs/data/functional/dict/"

    with open(start_path + 'user_to_book.json', 'rb') as f:
        user_to_book = pickle.load(f)

    with open(start_path + 'book_to_user.json', 'rb') as f:
        book_to_user = pickle.load(f)

    with open(start_path + 'user_book_to_rating.json', 'rb') as f:
        user_book_to_rating = pickle.load(f)

    book_translations_uu_old = {v: k for k, v in book_translations_old_uu.items()}
