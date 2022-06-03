import json
import os

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from gensim.models import LdaModel

from .algoritms.collaborative.user_user.predict import for_user
from django.forms.models import model_to_dict

from .api.models.status import create_status
from .api.recs import process_recs
from .models import Book
from .data.save_to_database import main
from .algoritms.content_based.lda import lda_train
from .algoritms.content_based.lda.lda_class import ContentBasedRecs
from .algoritms.hybrid.fwls import FeatureWeightedLinearStacking
from .algoritms.collaborative.user_user.uu_class import UserUserRecs
from .algoritms.collaborative.rbm.rbm_temp_class import RbmRecs
from .algoritms.collaborative.rbm import rbm_simple
import random
from .api import scores, user as user_req


def recommendations(request):
    try:
        count_param = request.GET["count"]
    except KeyError:
        count_param = None

    try:
        type_param = request.GET["type"]
    except KeyError:
        return JsonResponse(create_status(error="No type provided for recs"))

    return process_recs(100000, type_param, count_param)


def index(request):
    # lda_train.main()
    return HttpResponse("Ok")


def user_create_request(request):
    return user_req.create_user(request)


def user_get_request(request):
    return user_req.fetch_user()


def user_books_request(request):
    return user_req.books_user()


def set_score_request(request):
    return scores.set_score(request)


# def lda(request):
#     lda = LdaModel.load('./recs/algoritms/content_based/lda/static/model.lda')
#
#     for topic in lda.print_topics():
#         print("topic {}: {}".format(topic[0], topic[1]))
#
#     with open('./recs/static/lda.json') as json_file:
#         data = json.load(json_file)
#
#     context_dict = {
#         "topics": lda.print_topics(),
#         "number_of_topics": lda.num_topics,
#         "data_lda": data
#     }
#
#     print(os.getcwd())
#     return render(request, 'lda.html', context_dict)
#
#
# def json_request(request):
#     with open('./recs/static/lda.json') as json_file:
#         data = json.load(json_file)
#     return JsonResponse(data)
