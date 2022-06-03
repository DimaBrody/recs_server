import pandas as pd

books_df = pd.read_csv('books.csv', encoding='ISO-8859-1', on_bad_lines='skip', delimiter=";",
                       dtype={'ISBN': 'str', "Book-Title": 'str', "Book-Author": 'str',
                              "Year-Of-Publication": 'str', "Publisher": 'str',
                              "Image-URL-S": 'str', "Image-URL-M": 'str', "Image-URL-L": 'str'})
books_df.columns = ['isbn', "title", "author", "year", "publisher", "image_s", "image_m", "image_l"]
books_df = books_df.drop_duplicates(['title'])
books_df = books_df[books_df.year.str.len() <= 4]

books_ratings_df = pd.read_csv('books_ratings.csv', encoding='ISO-8859-1', on_bad_lines='skip', delimiter=";")
books_ratings_df.columns = ['userId', "isbn", "rating"]
books_ratings_df = books_ratings_df[books_ratings_df["rating"] > 0]
books_ratings_df = books_ratings_df[books_ratings_df.isbn.isin(set(books_df.isbn.values))]

unique_isbn_set = set(books_df.isbn.values)
isbnToBookId = {}
currentId = 0

for isbn in unique_isbn_set:
    isbnToBookId[isbn] = currentId
    currentId += 1

books_df['bookId'] = books_df.apply(lambda row: isbnToBookId[row.isbn], axis=1)
books_ratings_df["bookId"] = books_ratings_df.apply(lambda row: isbnToBookId[row.isbn], axis=1)
books_ratings_df = books_ratings_df.astype({'bookId': int})

books_cb_df = pd.read_csv('books_cb.csv')
books_cb_df.head()

titleToBookId = {}

bookIdToDesc = {}
bookIdToCategories = {}


def update_row(row):
    titleToBookId[row.title] = row.bookId
    return row


books_df.apply(update_row, axis=1)

books_cb_df['bookId'] = books_cb_df.isbn10.map(isbnToBookId)
books_cb_df['bookId'] = books_cb_df.bookId.fillna(books_cb_df.title.map(titleToBookId))

books_cb_df = books_cb_df.dropna(subset=["bookId"])
books_cb_df = books_cb_df.astype({'bookId': int})

books_cb_union_df = books_df[
    books_df.bookId.isin(set(books_cb_df.bookId.values))].copy()

for value in books_cb_df.values:
    bookIdToDesc[value[len(value) - 1]] = value[7]
    bookIdToCategories[value[len(value) - 1]] = value[5]

books_cb_union_df['categories'] = books_df.bookId.map(bookIdToCategories)
books_cb_union_df['description'] = books_df.bookId.map(bookIdToDesc)

books_cb_union_df = books_cb_union_df.dropna(subset=["categories", "description"])

books_cb_union_rating_df = books_ratings_df[books_ratings_df.bookId.isin(set(books_cb_union_df.bookId))]

books_ratings_df.to_csv("../edited/books-ratings-edited.csv", index=False)
books_df.to_csv("../edited/books-info-edited.csv", index=False)
books_cb_union_rating_df.to_csv("../edited/books-cb-rating-edited.csv", index=False)
books_cb_union_df.to_csv("../edited/books-cb-edited.csv", index=False)
