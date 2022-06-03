from django.http import HttpResponse, JsonResponse

from recs.api.models.status import create_status
from recs.api.utils import get_body
from recs.models import Rating, User


def set_score(request):
    body = get_body(request)

    Rating.objects.create(user_id=body["user_id"], book_id=body["book_id"], rating=body["rating"])

    return JsonResponse(
        create_status(f"""Score {body["rating"]} for book with id {body["book_id"]} successfully created"""))
