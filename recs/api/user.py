from django.forms import model_to_dict
from django.http import JsonResponse

from recs.api.models.status import create_status
from recs.api.utils import get_body
from recs.models import User, Book, Rating

DEFAULT_ID = 100000


def create_user(request):
    body = get_body(request)

    User.objects.create(id=DEFAULT_ID, username=body["username"])

    return JsonResponse(create_status(f'User \'' + body["username"] + '\' created successfully'))


def fetch_user():
    user = model_to_dict(User.objects.get(id=DEFAULT_ID))

    return JsonResponse(user)


def book_id_to_model(book_id, rating):
    dict_book = model_to_dict(Book.objects.get(id=book_id))
    dict_book["rating"] = rating
    return dict_book


def books_user():
    ratings = [model_to_dict(rating) for rating in Rating.objects.filter(user_id=DEFAULT_ID)]
    books = [book_id_to_model(rating["book_id"], rating["rating"]) for rating in ratings]

    return JsonResponse(create_status(content={"title": "Оцінені книги", "books": books}))
