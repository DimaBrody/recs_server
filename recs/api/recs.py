from django.http import JsonResponse

from recs.algoritms.collaborative.matrix_factorization.mf_class import MatrixFactorizationRecs
from recs.algoritms.collaborative.rbm.rbm_temp_class import RbmRecs
from recs.algoritms.collaborative.user_user.uu_class import UserUserRecs
from recs.algoritms.content_based.lda.lda_class import ContentBasedRecs
from recs.algoritms.global_recs.global_recs import GlobalRecs
from recs.algoritms.hybrid.fwls import FeatureWeightedLinearStacking
from recs.api.models.status import create_status

DEFAULT_COUNT = 10


def process_recs(user_id, recs_type, count=None):
    try:
        recs = classes_dict[recs_type]()
    except KeyError:
        return JsonResponse(create_status(error=f"""Type "{recs_type}" does not exist"""))

    recommended = recs.recommend_items(user_id, int(count) if count else DEFAULT_COUNT)
    print(recommended)
    return JsonResponse(recs.convert_to_response(recommended))


class FwlsRecs:
    pass


classes_dict = {
    "base": UserUserRecs,
    "global": GlobalRecs,
    "matrix": MatrixFactorizationRecs,
    "rbm": RbmRecs,
    "cb": ContentBasedRecs,
    "hybrid": FeatureWeightedLinearStacking
}
