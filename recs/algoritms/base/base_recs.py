from abc import ABCMeta, abstractmethod

from django.forms import model_to_dict

from recs.api.models.status import create_status
from recs.models import Book


class BaseRecs(metaclass=ABCMeta):
    @abstractmethod
    def predict_score(self, user_id, item_id):
        pass

    @abstractmethod
    def recommend_items(self, user_id, num=6):
        pass

    @abstractmethod
    def get_title(self):
        pass

    def convert_to_response(self, recommended_items):
        recommended_books = [(k, model_to_dict(Book.objects.get(id=k))) for k, _ in recommended_items]

        return create_status(content={"title": self.get_title(), "books": [b for _, b in recommended_books]})
