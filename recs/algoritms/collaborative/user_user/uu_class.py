from ...base.base_recs import BaseRecs

from .predict import for_user, predict


class UserUserRecs(BaseRecs):

    def predict_score(self, user_id, item_id):
        return predict(user_id, item_id)

    def recommend_items(self, user_id, num=6):
        return for_user(user_id, num)

    def get_title(self):
        return "Базова КФ"
