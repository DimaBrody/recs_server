from django.core.validators import validate_comma_separated_integer_list
from django.db import models


class User(models.Model):
    id = models.IntegerField(unique=True, primary_key=True)
    username = models.CharField(max_length=255)


class Rating(models.Model):
    user_id = models.CharField(max_length=16)
    book_id = models.CharField(max_length=16)
    rating = models.DecimalField(decimal_places=2, max_digits=4)
    type = models.CharField(max_length=8, default='explicit')

    def __str__(self):
        return "user_id: {}, book_id: {}, rating: {}, type: {}" \
            .format(self.user_id, self.book_id, self.rating, self.type)


class Book(models.Model):
    id = models.IntegerField(unique=True, primary_key=True)
    isbn = models.CharField(max_length=13)
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    year = models.IntegerField()
    image_s = models.CharField(max_length=2083)
    image_m = models.CharField(max_length=2083)
    image_l = models.CharField(max_length=2083)
    publisher = models.CharField(max_length=255)


class BookCB(models.Model):
    id = models.IntegerField(unique=True, primary_key=True)
    isbn = models.CharField(max_length=13)
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    publisher = models.CharField(max_length=255)
    description = models.TextField(max_length=511)
    genres = models.TextField(max_length=255)
    lda_vector = models.CharField(max_length=56, null=True)
    sim_list = models.CharField(max_length=512, default='')


class LdaSimilarity(models.Model):
    source = models.CharField(max_length=16, db_index=True)
    target = models.CharField(max_length=16)
    similarity = models.DecimalField(max_digits=8, decimal_places=7)

    class Meta:
        db_table = 'lda_similarity'

    def __str__(self):
        return "[({} => {}) sim = {}]".format(self.source,
                                              self.target,
                                              self.similarity)
