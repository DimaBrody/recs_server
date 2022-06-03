import pandas as pd
import os
from recs.models import Book, BookCB


def save_books_info(df):
    len_df = len(df)
    books = []
    for index, row in df.iterrows():
        books.append(Book(id=row["bookId"], isbn=row["isbn"], title=row["title"], author=row["author"],
                          year=row["year"], publisher=row["publisher"], image_s=row["image_s"],
                          image_m=row["image_m"], image_l=row["image_l"]))

        if index % 5000 == 0:
            print("process: {:.2f}%".format((index / len_df) * 100))

    print("saving:")
    Book.objects.bulk_create(books)


duplicates = {}


def save_books_cb(df):
    len_df = len(df)
    books = []
    #   description = models.TextField(max_length=511)
    #     genres = models.TextField(max_length=255)
    for index, row in df.iterrows():
        shorten_title = row["title"][:25]
        if not duplicates.get(shorten_title):
            books.append(BookCB(id=row["bookId"], title=row["title"], author=row["author"], publisher=row["publisher"],
                                description=row["description"],
                                genres=row["categories"]))
            duplicates[shorten_title] = True

        if index % 5000 == 0:
            print("process: {:.2f}%".format((index / len_df) * 100))

    print("saving:")
    BookCB.objects.bulk_create(books)


# books_df.columns = ['isbn', "title", "author", "year", "publisher", "image_s", "image_m", "image_l"]

# id = models.IntegerField(unique=True, primary_key=True)
#    isbn = models.CharField(max_length=13)
#    name = models.CharField(max_length=255)
#    author = models.CharField(max_length=255)
#    year = models.IntegerField()
#    image_s = models.CharField(max_length=2083)
#    image_m = models.CharField(max_length=2083)
#    image_l = models.CharField(max_length=2083)
#    publisher = models.CharField(max_length=255)
def main():
    print(os.getcwd())
    # df_info = pd.read_csv("./recs/data/edited/books-info-edited.csv")

    save_books_info(pd.read_csv("./recs/data/edited/books-info-edited.csv"))
    save_books_cb(pd.read_csv("./recs/data/edited/books-cb-edited.csv"))

    pass


if __name__ == "__main__":
    main()
