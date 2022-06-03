from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('user/create', views.user_create_request),
    path('user', views.user_get_request),
    path('user/books', views.user_books_request),
    path('score', views.set_score_request),
    path('recommendations', views.recommendations),
    path('lda', views.lda),
    path('json', views.json_request),
]
